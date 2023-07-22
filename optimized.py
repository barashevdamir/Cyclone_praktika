import cv2
import numpy as np
import os
import time


def read_images_and_times():
    # Путь к папке с изображениями
    images_dir = 'Bali-Tukad-Cepung-Waterfall'

    # Получаем список всех .jpg файлов в папке
    filenames = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if
                 os.path.isfile(os.path.join(images_dir, f)) and f.endswith('.jpg')]

    # Упорядочиваем файлы так, чтобы _under был первым, _over был последним
    filenames.sort(key=lambda x: ('_over' in x, '_under' not in x, x))

    # Соответствующее время экспозиции в секундах от наименьшего к наибольшему
    times = np.array([1/6.0, 1.3, 5.0], dtype=np.float32)

    images = []
    for filename in filenames:
        im = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        if im is None:
            print(f"Error: {filename} is not a valid image file.")
            continue
        images.append(im)

    if not images:
        print("No valid images to process. Exiting.")
        exit()

    return images, times


def process_images(images, times):
    # Алигнирование
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

    # Получаем функцию отклика камеры (CRF)
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, times)

    # Объединение изображения в линейное изображение HDR
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times, responseDebevec)

    # Отображение тонов с использованием метода Драго для получения 24-битного цветного изображения
    start_time_drago = time.perf_counter()
    gamma = 0.6
    saturation = 0.4
    bias = 0.9
    tonemapDrago = cv2.createTonemapDrago(gamma, saturation, bias)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 3 * ldrDrago
    end_time_drago = time.perf_counter()
    print(f"Метод Drago выполнялся {end_time_drago - start_time_drago} секунд")

    # Отображение тонов с использованием метода Рейнхарда для получения 24-битного цветного изображения
    start_time_reinhard = time.perf_counter()
    gamma = 1.5
    intensity = 0.0
    light_adapt = 0.0
    color_adapt = 0.0
    tonemapReinhard = cv2.createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)
    ldrReinhard = tonemapReinhard.process(hdrDebevec)
    end_time_reinhard = time.perf_counter()
    print(f"Метод Reinhard выполнялся {end_time_reinhard - start_time_reinhard} секунд")

    # Отображение тонов с использованием метода Мантюка для получения 24-битного цветного изображения
    start_time_mantiuk = time.perf_counter()
    gamma = 1.3
    scale = 0.85
    saturation = 0.9
    tonemapMantiuk = cv2.createTonemapMantiuk(gamma, scale, saturation)
    ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
    ldrMantiuk = 3 * ldrMantiuk
    end_time_mantiuk = time.perf_counter()
    print(f"Метод Mantiuk выполнялся {end_time_mantiuk - start_time_mantiuk} секунд")

    return hdrDebevec, ldrDrago, ldrReinhard, ldrMantiuk


def save_images(hdrDebevec, ldrDrago, ldrReinhard, ldrMantiuk):
    # Указываем путь к папке, в которую хотим сохранить результаты
    results_dir = 'results'

    # Проверяем, существует ли папка, если нет, то создаем
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Сохраняем результаты
    cv2.imwrite(os.path.join(results_dir, "hdrDebevec.hdr"), hdrDebevec)
    cv2.imwrite(os.path.join(results_dir, "ldr-Drago.jpg"), cv2.cvtColor(ldrDrago * 255, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(results_dir, "ldr-Reinhard.jpg"), cv2.cvtColor(ldrReinhard * 255, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(results_dir, "ldr-Mantiuk.jpg"), cv2.cvtColor(ldrMantiuk * 255, cv2.COLOR_RGB2BGR))


def main():
    images, times = read_images_and_times()
    hdrDebevec, ldrDrago, ldrReinhard, ldrMantiuk = process_images(images, times)
    save_images(hdrDebevec, ldrDrago, ldrReinhard, ldrMantiuk)


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Программа выполнялась {execution_time} секунд")
