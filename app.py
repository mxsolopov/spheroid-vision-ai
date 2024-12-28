import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, UpSampling2D, 
    concatenate, Input, Conv2DTranspose
)
from tensorflow.keras.models import Model
import cv2
from PIL import Image
import io
import pandas as pd

# Константы
IMAGE_SIZE = 224
PATCH_SIZE = 224
OVERLAP = 32

@st.cache_resource
def load_model():
    """Загрузка и кэширование модели"""
    def unet_model(input_size=(IMAGE_SIZE, IMAGE_SIZE, 3)):
        inputs = Input(input_size)
        
        # Base model VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

        # Encoder
        conv1 = base_model.get_layer('block1_conv2').output
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = base_model.get_layer('block2_conv2').output
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = base_model.get_layer('block3_conv3').output
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = base_model.get_layer('block4_conv3').output
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = base_model.get_layer('block5_conv3').output

        # Decoder
        up6 = UpSampling2D(size=(2, 2))(conv5)
        up6 = concatenate([up6, conv4], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        up7 = concatenate([up7, conv3], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up8 = UpSampling2D(size=(2, 2))(conv7)
        up8 = concatenate([up8, conv2], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        up9 = concatenate([up9, conv1], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])
        
        # Загрузка весов
        model.load_weights('models/unet_weights.h5')
        return model
    
    return unet_model()

def split_into_patches(image, patch_size=224, overlap=32):
    """Разбивает изображение на патчи с перекрытием"""
    patches = []
    h, w, _ = image.shape
    
    h_steps = max(1, int(np.ceil((h - patch_size) / (patch_size - overlap))) + 1)
    w_steps = max(1, int(np.ceil((w - patch_size) / (patch_size - overlap))) + 1)
    
    for i in range(h_steps):
        for j in range(w_steps):
            y_start = min(i * (patch_size - overlap), h - patch_size)
            x_start = min(j * (patch_size - overlap), w - patch_size)
            
            patch = image[y_start:y_start+patch_size, x_start:x_start+patch_size]
            patches.append((patch, y_start, x_start))
    
    return patches

def reconstruct_image(patches, image_shape, patch_size=224, overlap=32):
    """Восстанавливает изображение из патчей"""
    h, w = image_shape[:2]
    reconstructed = np.zeros(image_shape)
    weights = np.zeros(image_shape[:2] + (1,))
    
    weight_kernel = np.ones((patch_size, patch_size, 1))
    if overlap > 0:
        for i in range(overlap):
            weight_kernel[i, :, 0] *= (i / overlap)
            weight_kernel[-i-1, :, 0] *= (i / overlap)
            weight_kernel[:, i, 0] *= (i / overlap)
            weight_kernel[:, -i-1, 0] *= (i / overlap)
    
    for patch, y_start, x_start in patches:
        y_end = min(y_start + patch_size, h)
        x_end = min(x_start + patch_size, w)
        patch_h = y_end - y_start
        patch_w = x_end - x_start
        
        current_weight = weight_kernel[:patch_h, :patch_w]
        reconstructed[y_start:y_end, x_start:x_end] += patch[:patch_h, :patch_w] * current_weight
        weights[y_start:y_end, x_start:x_end] += current_weight
    
    mask = weights > 0
    reconstructed[mask] /= weights[mask]
    
    return reconstructed

def process_image(image, model, threshold=0.5):
    """Обработка изображения: разбиение на патчи, предсказание и сборка"""
    # Разбиение на патчи
    patches = split_into_patches(image, PATCH_SIZE, OVERLAP)
    total_patches = len(patches)
    
    # Создаем прогресс-бар
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Предсказание для каждого патча
    predicted_patches = []
    for idx, (patch, i, j) in enumerate(patches, 1):
        # Обновляем статус
        status_text.text(f'Обработка патча {idx} из {total_patches}')
        progress_bar.progress(idx/total_patches)
        
        patch_input = np.expand_dims(patch, axis=0)
        pred = model.predict(patch_input, verbose=0)[0]  # отключаем вывод прогресса TensorFlow
        
        predicted_patches.append((pred, i, j))
    
    # Очищаем прогресс-бар и статус после завершения
    progress_bar.empty()
    status_text.empty()
    
    # Восстановление полной маски
    mask_shape = image.shape[:2] + (1,)
    reconstructed_mask = reconstruct_image(predicted_patches, mask_shape, PATCH_SIZE, OVERLAP)
    
    # Отладочная информация о реконструированной маске
    st.write(f"Реконструированная маска: "
            f"мин={reconstructed_mask.min():.3f}, "
            f"макс={reconstructed_mask.max():.3f}, "
            f"среднее={reconstructed_mask.mean():.3f}")
    
    # Бинаризация маски
    binary_mask = (reconstructed_mask > threshold).astype(np.uint8)
    
    return binary_mask

def main():
    st.title("Сегментация сфероидов")
    st.write("Загрузите изображения для сегментации сфероидов")
    
    # Загрузка модели
    model = load_model()
    
    # Загрузка нескольких изображений
    uploaded_files = st.file_uploader("Выберите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        # Добавляем слайдер для порога
        threshold = st.slider("Порог бинаризации", 0.0, 1.0, 0.5, 0.01)
        
        if st.button("Выполнить сегментацию"):
            # Создаем DataFrame для хранения данных всех изображений
            all_data = []
            
            # Обрабатываем каждое изображение
            for file_idx, uploaded_file in enumerate(uploaded_files, 1):
                st.write(f"### Обработка изображения {file_idx}: {uploaded_file.name}")
                
                # Преобразование загруженного файла в изображение
                image = Image.open(uploaded_file)
                image = np.array(image)
                
                # Нормализация изображения
                image = image.astype(float) / 255.0
                
                st.image(uploaded_file, caption=f"Исходное изображение {file_idx}", use_column_width=True)
                
                with st.spinner(f"Выполняется сегментация изображения {file_idx}..."):
                    # Обработка изображения
                    mask = process_image(image, model, threshold)
                    
                    # Преобразование маски для отображения (0-255)
                    display_mask = mask * 255
                    
                    # Отображение результата
                    st.image(display_mask, caption=f"Результат сегментации {file_idx}", use_column_width=True)
                    
                    # Наложение маски на исходное изображение с метками площади
                    overlay = image.copy()
                    mask = np.squeeze(mask)
                    
                    # Получаем свойства каждого объекта
                    num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                    
                    # Для каждого сфероида (пропускаем фон с индексом 0)
                    for i in range(1, num_regions):
                        # Получаем площадь
                        area = stats[i, cv2.CC_STAT_AREA]
                        # Получаем центроид (центр масс)
                        center_x = int(centroids[i][0])
                        center_y = int(centroids[i][1])
                        
                        # Окрашиваем сфероид в красный
                        overlay[labels == i] = [1, 0, 0]
                        
                        # Добавляем текстовую метку с площадью
                        img_for_text = (overlay * 255).astype(np.uint8)
                        cv2.putText(img_for_text, 
                                  f'Area: {area}px', 
                                  (center_x, center_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5,
                                  (255, 255, 255),
                                  2)
                        overlay = img_for_text.astype(float) / 255.0
                        
                        # Добавляем данные в общий список
                        all_data.append({
                            'Имя файла': uploaded_file.name,
                            'ID изображения': file_idx,
                            'ID сфероида': i,
                            'Площадь (пиксели)': area
                        })
                    
                    st.image(overlay, caption=f"Наложение маски на изображение {file_idx}", use_column_width=True)
            
            # Создаем DataFrame со всеми данными
            df = pd.DataFrame(all_data)
            
            # Отображаем сводную таблицу
            st.write("### Сводная таблица всех сфероидов:")
            st.dataframe(df)
            
            # Добавляем статистику по изображениям
            stats_by_image = df.groupby('Имя файла').agg({
                'Площадь (пиксели)': ['count', 'mean', 'min', 'max']
            }).round(2)
            stats_by_image.columns = ['Количество сфероидов', 'Средняя площадь', 'Минимальная площадь', 'Максимальная площадь']
            
            st.write("### Статистика по изображениям:")
            st.dataframe(stats_by_image)
            
            # Кнопки для скачивания данных
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать все данные в CSV",
                data=csv,
                file_name="all_spheroid_data.csv",
                mime="text/csv"
            )
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Все данные', index=False)
                stats_by_image.to_excel(writer, sheet_name='Статистика')
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="Скачать все данные в Excel",
                data=excel_data,
                file_name="all_spheroid_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()