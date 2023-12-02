import cv2
import os
import datetime
import time
# import ModulKlasifikasiCitraCNN as mCNN
import numpy as np

# Untuk penamaan semua class di Model ML
cardName = [
    "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "10C", "JC", "QC", "KC", "AC",
    "2H", "3H", "4H", "5H", "6H", "7H", "8H", "9H", "10H", "JH", "QH", "KH", "AH",
    "2S", "3S", "4S", "5S", "6S", "7S", "8S", "9S", "10S", "JS", "QS", "KS", "AS",
    "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "10D", "JD", "QD", "KD", "AD",
    "Joker"
]

# Kita mulai dari index 0
cardNameIndex = 0

# Fungsi dari pak Akok buat penamaan file menurut waktu diambil
def GetFileName():
    x = datetime.datetime.now()
    s = x.strftime('%Y-%m-%d-%H%M%S%f')
    return s

# Fungsi dari ChatGPT buat menentukan luas area dari 4 titik
def polygon_area(points):
    # 'points' adlh input berupa array yg berisikan 4 titik koordinat kartu
    points = np.vstack((points, points[0]))
    area = 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))
    return area

# Fungsi dari pak Akok
def CreateDir(path):
    ls = []
    head_tail = os.path.split(path)
    ls.append(path)
    while len(head_tail[1])>0:
        head_tail = os.path.split(path)
        path = head_tail[0]
        ls.append(path)
        head_tail = os.path.split(path)
    for i in range(len(ls)-2,-1,-1):
        sf = ls[i]
        isExist = os.path.exists(sf)
        if not isExist:
            os.makedirs(sf)

# Fungsi dari pak Akok juga, tapi harus kita modif
def CreateDataSet(sDirektoriData,sKelas,NoKamera,FrameRate):
    global cardName, cardNameIndex

    # For webcam input:
    cap = cv2.VideoCapture(0)
    ip = 'http://192.168.100.172:8080/video'
    cap.open(ip)
    TimeStart = time.time()

    # Ini buat ngelimit 1 data kartu ambil berapa detik
    saveTimeLimit = time.time()

    # For start taking pics
    # Variable Buat nandain bahwa frame video itu direcord
    isSaving = False

    while cap.isOpened():
        success, frame = cap.read()
        
        # Buat dulu folder sesuai dengan datasetnya
        sDirektoriKelas = sDirektoriData+"/"+cardName[cardNameIndex]
        CreateDir(sDirektoriKelas)

        if not success:
            print("Ignoring empty camera frame.")
            continue

        isDetected = False
        



        # DI SINI LAH IDE DATA SET BISA BERBEDA2

        # Ini cuma sebagai contoh aja ya, bisa kalian modif biar nggak sama tiap2 orang
        # Kemiripan dan penjelasan saat demo ditanggung penumpang :)
        
        # Cara yg kupake:
        # 1. Grayscale dulu
        # 2. Di threshold (adaptiveThreshold) biar cuma dapet pola kartunya aja
        # 3. Di crop, baru fotonya disimpen buat dijadiin dataset

        # ^ cuma contoh aja. Bisa kalian bedain sesuai selera
        # TAPI PASTIIN DATASET YG DIDAPAT ITU:
        # 1. Nggak ada object penghalang (tangan, benda lain)
        # 2. Backgroundnya harus sama (Kalo gambarnya berwarna, ya backgroundnya kalo ijo harus ijo semua)
        # 3. Sebisa mungkin datasetnya itu hanya memuat gambar yg diperlukan
            # Sebagai contoh di crop kartunya aja, gausah ada background
            # Background kalo sama di setiap kartu gpp
            # Tapi kalo udah beda2, itu bikin Modelnya pusing
        # 4. Pastiin fotonya cukup high res
        
        
        
        
        # Image processing
        # ======= 1. ======= Pertama, gambar kita buat grayscale dulu
        # Gray
        imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("1. Gray scale dulu", imGray)

        # ======= 2. ======= Kedua, kita threshold buat ambil
        # Threshold
        #                                                    Nilai 71 dan 10 bisa diatur sesuai kebutuhan masing2. Caranya? Cari aja di google 71 itu apa 10 itu apa. Kalo udah coba2 nilai yg pas buat kamera dan kartu kalian
        # NOTES:                                                                                        v   v    
        imThres = cv2.adaptiveThreshold(imGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,10)
        # cv2.imshow("2. Adaptive thres", imThres)

        # ======= 3. ======= Next, kita ambil component yg connected.
        # Ini diambil dari contoh nya pak Akok di catatan buat HSV, cuma diedit dikit2
        # https://drive.google.com/file/d/1nuPMCajNSBYXBYO4t5nESIi74eNIzb1b/view
        # Connecting
        totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(imThres, 4, cv2.CV_32S)
        # Big index adalah array untuk menampung index mana yg luas area connected componentnya sesuai keinginan kita
        bigIndex = []
        for i in range(totalLabels):
            hw = values[i,2:4]
            # 100 dan 300 itu untuk cari widht diantara 100-300, 
            # 300 dan 500 itu cari height antara 300-500
            # Harus dicoba2 biar hasilnya sesuai dengan kartu kalian
            # Cara nyoba gimana? Kalo gambar kartunya belom di kotakin, berarti nilainya masih salah. Tweeking aja coba2
            if (100<hw[0]<900 and 300<hw[1]<1000):
                bigIndex.append(i)

        # ======= 4. ======= Check, apakah ada connected component yg sesuai dengan luas yg kita define
        # Kalo ada kita kotakin trus kita kotakin
        for i in bigIndex:
            topLeft = values[i,0:2]
            bottomRight = values[i,0:2]+values[i,2:4]
            # v                     v       Disini aku ngotakin di gambar asli 'frame'
            frame = cv2.rectangle(frame, topLeft, bottomRight, color=(0,0,255), thickness=3)

            # Disini ada break, yg berarti kita cuma ngambil 1 item doang
            break
        # Trus tampilin
        # cv2.imshow("4. Hasil habis dikotakin", frame)
        
        # ======= 5. ======= Kita crop yg dikotakin tadi
        for i in bigIndex:
            topLeft = values[i,0:2]
            bottomRight = values[i,0:2]+values[i,2:4]
            
            # Ini buat ngecrop gambarnya
            cardImage = imThres[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
            
            # Lanjut ditampilin
            cv2.imshow('5. Hasil dari cardImage', cardImage)
            # Gambar ini yg bakal disimpen dan masuk jadi dataset model
            
            # Lagi lagi cuma ngambil 1 item doang
            break
        
        # ======= 6. ======= Ini buat ngerecord datasetnya.
        # Diambil juga dari modul bapaknya, yg dimodif dikit
        TimeNow = time.time()
        if TimeNow-TimeStart>1/FrameRate:
            sfFile = sDirektoriKelas+"/"+GetFileName()
            # Kita bakal nyimpen kalo sudah teken spasi, dan kal0 ada kartu yg terdeteksi
            if isSaving and len(bigIndex) > 0:
                cv2.imwrite(sfFile+'.jpg', cardImage)
            TimeStart = TimeNow

        # Buat Kata2 doang
        cv2.putText(frame, "Nama Kartu yg direkam:", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, f"{cardNameIndex+1}. " + cardName[cardNameIndex], (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Tekan spasi untuk mulai record", (0, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Buat visualisasi record
        if isSaving:
            cv2.putText(frame, "Record", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            saveTimeLimit = time.time()

        cv2.imshow("Tampilan akhir", frame)
        
        key = cv2.waitKey(5)

        # Trigger tekan spasi untuk mulai menyimpan gambar
        if key == 32:
            isSaving = not isSaving

        # Kalo udah lebih dari 5 detik, penyimpanan foto selesai
        # Bisa edit sesuai kebutuhan brp detiknya
        if key == ord('s'):
            cardNameIndex += 1
            isSaving = False

        if key & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# Nama Parent Folder dimana dataset akan ditaruh
DirektoriDataSet = "dataset"

# Kita panggil fungsinya sekali aja, nanti akan looping sendiri
CreateDataSet(DirektoriDataSet, "Kosong Aja udah ini, nanti kan di replace", NoKamera=0, FrameRate=20)