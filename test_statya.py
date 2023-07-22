import cv2
import numpy as np
import time


def readImagesAndTimes():
    times = np.array([1/8.0, 1/5.0, 0.8], dtype=np.float32)

    filenames = ["Tree/tree_under.jpg", 'Tree/tree.jpg', 'Tree/tree_over.jpg']

    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)

    return images, times


if __name__ == '__main__':

    start_time = time.perf_counter()

    # Считывание изображений и времени экспозиции
    print("Reading images ... ")

    images, times = readImagesAndTimes()

    # Выравниваем входные изображения
    print("Aligning images ... ")
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

    # Получаем функцию отклика камеры (CRF)
    print("Calculating Camera Response Function (CRF) ... ")
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, times)

    # Объединяем изображения в линейное изображение HDR
    print("Merging images into one HDR image ... ")
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
    # Сохраняем HDR-изображение
    cv2.imwrite("hdrDebevec.hdr", hdrDebevec)
    print("saved hdrDebevec.hdr ")

    # Тональная карта с использованием метода Драго для получения 24-битного цветного изображения
    print("Tonemaping using Drago's method ... ")
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 3 * ldrDrago
    cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)
    print("saved ldr-Drago.jpg")

    # # Тональная карта с использованием метода Дюрана для получения 24-битного цветного изображения
    # print("Tonemaping using Durand's method ... ")
    # tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
    # ldrDurand = tonemapDurand.process(hdrDebevec)
    # ldrDurand = 3 * ldrDurand
    # cv2.imwrite("ldr-Durand.jpg", ldrDurand * 255)
    # print("saved ldr-Durand.jpg")

    # Тональная карта с использованием метода Рейнхарда для получения 24-битного цветного изображения
    print("Tonemaping using Reinhard's method ... ")
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(hdrDebevec)
    cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)
    print("saved ldr-Reinhard.jpg")

    # Тональная карта с использованием метода Мантюка для получения 24-битного цветного изображения
    print("Tonemaping using Mantiuk's method ... ")
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
    ldrMantiuk = 3 * ldrMantiuk
    cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)
    print("saved ldr-Mantiuk.jpg")

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"Программа выполнялась {execution_time} секунд")
