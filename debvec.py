import cv2
import numpy as np
import os
import time
import concurrent.futures

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


def tonemap(tonemap_create_method, hdrDebevec, *args):
    start_time = time.perf_counter()
    tonemap_method = tonemap_create_method(*args)
    ldr = tonemap_method.process(hdrDebevec)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Метод {tonemap_create_method.__name__} выполнялся {execution_time} секунд")
    return ldr



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

    ldrDrago = None
    ldrReinhard = None
    ldrMantiuk = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(tonemap, method, hdrDebevec, *params): (method, params)
            for method, params in [
                (cv2.createTonemapDrago, (0.6, 0.4, 0.9)),
                (cv2.createTonemapReinhard, (1.5, 0.0, 0.0, 0.0)),
                (cv2.createTonemapMantiuk, (1.3, 0.85, 0.9))
            ]
        }

        for future in concurrent.futures.as_completed(futures):
            method, params = futures[future]
            try:
                result = future.result()
                if method == cv2.createTonemapDrago:
                    ldrDrago = 3 * result
                elif method == cv2.createTonemapReinhard:
                    ldrReinhard = result
                elif method == cv2.createTonemapMantiuk:
                    ldrMantiuk = 3 * result
            except Exception as exc:
                print(f'Tonemap {method.__name__} generated an exception: {exc}')

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
