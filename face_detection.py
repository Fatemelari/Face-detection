import cv2                                                                    #opencvکردن کتابخانه import
import numpy as np                                                            #numpyکردن کتابخانه import


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_defaul.xml')    #xmlذخیره کردن الگوریتم تشخیص چهره با فرمت
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')                    #xmlذخیره کردن الگوریتم تشخیص چشم با فرمت
lip_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')                  #xmlذخیره کردن الگوریتم تشخیص دهان با فرمت

                                                

vc=cv2.VideoCapture(0)                                                        #فعال کردن وبکم_در صورتی ک ب چند وبکم دسترسی داشتید میتوانید با تغیر شماره داخل پرانتز وبکم مورد نظر را فعال کنید_

while True:                                                                   #ویدیو مجوعه ای از عکسهاست ک پشت سر هم نمایش داده میشوند.و برای پردازش یک ویدیو لازم است تمام ان عکس ها پردازش شوند. 
                                                                              #پس با ایجاد یک حلقه بی نهایت تمام این عکس های ورودی را پردازش می کنیم.تا کابر با زدن کلید کیو:| دسترسی ب وکم را قطع کند و دیگر ورودی دریافت نشود
       ret,img = vc.read()                                                    #خواندن تصاویر از وبکم
       gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)                         #graylevelتبدیل عکس ها به 

       faces = face_cascade.detectMultiScale(gray, 1.3, 5)                    #فراخوانی الگوریتم تشخیص چهره
       for (x,y,w,h) in faces:                                                #برای شناسایی تمام صورت های داخل تصویرforایجاد حلقه 
            
            center = (x + w//2, y + h//2)                                     #|
            m=w//2                                                            #|=>  رسم دایره ابی دور صورت های تشخیص داده شده
            cv2.circle(img, center,m,(255, 0,0), 2)                           #|
            
            mat=cv2.GaussianBlur(img,(15,15),0)                               #|
            mask1=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)        #|      کل تصویر را مات کنیم. gaussian.blur برای مات کردن یک قسمت از تصویرابتدا باید،با استفاده از تابع
            mask1=cv2.circle(mask1,center,m,255,-1)                           #|=>    و ناحیه ای که می خواهیم مات نشود را در ارایه ای ک با ابعاد تصویر اصلی ایجاد کردیم مشخص کنیم_ماسک کردن_.ک در اینجا همان دایره محاط شده ب صورت هاست
            l=cv2.bitwise_and(img,img,mask=mask1)                             #|      تصویر اصلی و تصویر ماسک شده را با هم ادغام میکینم.تا دایره ی محاط شده ب چهره را،ساده وبدون مات شدن داشته باشیم. bitwise_and سپس با استفاده از تابع
            z=cv2.bitwise_or(l,mat)                                           #|     تصویر بدست امده در مرحله قبل را با تصویر اصلی ک مات شده بودادغام می کینم.تا همه جای تصویر ب جز قسمت صورت مات شده باشدbitwise_or و در اخر با استفاده از تابع 
            cv2.imshow('blur',z)                                              #|
             

            roi_gray = gray[y:y+h, x:x+w]                                     #|محدود کردن تصویر ب یک مستطیل محاط شده به چهره ب صورت خاکستری و رنگی
            roi_color = img[y:y+h, x:x+w]                                     #|زیرا در ادامه دستورات ب شکلی هست ک نیازی ب پردازش بقیه تصویر نداریم.و تمامی دستورات باید در همین محدوداجرا شوند
            roi_color2=roi_color.copy()                                       #|ایجاد یک کپی از تصویر رنگی محدود شده ب صورت
           
            mask=np.zeros(roi_gray.shape[:2],dtype='uint8')                   #|
            eyes = eye_cascade.detectMultiScale(roi_gray,3,7)                 #|     فراخوانی الگوریتم تشخیص چشم  
            for (ex,ey,ew,eh) in eyes:                                        #|     برا شناسایی تمام چشم های موجود در تصویرforو ایجاد حلقه 
               eye =cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)#|=>   کشیدن مستطیل دور چشم های تشخیص داده شده
               c=cv2.rectangle(mask,(ex,ey),(ex+ew,ey+eh),255,-1)             #|     ایجاد یک ارایه با ابعاد تصویر محدود شده ب صورت، ب منظور ماسک کردن چشم ها 
               masked=cv2.bitwise_and(roi_color,roi_color,mask=mask)          #|     bitwise_andو استخراج انها از تصویر با تابع
               cv2.imshow('eye',masked)                                       #|     و نمایش انها در ارایه کنار تصویر اصلی
               
               
               c=ex+ew//2,ey+eh//2                                            #|
               cv2.circle(roi_color2,c,20,(255,255,255),-1)                   #|
               cv2.circle(roi_color2,c,18,(0,0,0),-1)                         #|
               cv2.circle(roi_color2,c,16,(255,255,255),-1)                   #|
               cv2.circle(roi_color2,c,14,(0,0,0),-1)                         #|
               cv2.circle(roi_color2,c,12,(255,255,255),-1)                   #|=>    رسم دایره های تو در تو برای قرار گیری ب جای چشم اصلی
               cv2.circle(roi_color2,c,10,(0,0,0),-1)                         #|
               cv2.circle(roi_color2,c,8,(255,255,255),-1)                    #|
               cv2.circle(roi_color2,c,6,(0,0,0),-1)                          #|
               cv2.circle(roi_color2,c,4,(255,255,255),-1)                    #|
               cv2.circle(roi_color2,c,2,(0,0,0),-1)                          #|
               cv2.imshow('hipnotism',roi_color2)                             #|
               
               
               

            masklip=np.zeros(roi_gray.shape[:2],dtype='uint8')                #|         
            lip=lip_cascade.detectMultiScale(roi_gray,4,5)                    #|    فراخوانی الگوریتم تشخیص دهان 
            for (mx,my,mw,mh) in lip:                                         #|    برای شناسایی تمام دهن های موجود در تصویرfor و ایجاد حلقه 
                cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)    #|=>  کشیدن مستطیل دور دهان تشخیص داده شده 
                cv2.rectangle(masklip,(mx,my),(mx+mw,my+mh),255,-1)           #|    ایجاد یک ارایه با ابعاد تصویر محدود شده ب صورت، ب منظور ماسک کردن دهان ها
                maskk=cv2.bitwise_and(roi_color,roi_color,mask=masklip)       #|    bitwise_andو استخراج انها از تصویر با تابع
                cv2.imshow('lips',maskk)                                      #|    و نمایش انها در ارایه کنار تصویر اصلی
                
                
       cv2.imshow('video',img)                                                # نمایش تصاویر پردازش شده ب صورت ویدیو
       if cv2.waitKey(1) & 0xff==ord('q'):                                    #را بزند دسترسی ب وبکم قطع میشود و دیگر تصویری پردازش  و نمایش داده نمی شودqدر صورتی ک کاربر کلید 
           break
     
vc.release()                                                                  #این تابع نشان می دهد دیگر تصویری گرفته نخواهد شد و وبکم خاموش است
cv2.destroyAllWindows()                                                       #از بین بردن پنجره های ایجاد شده ب نظور ازاد کردن فضای اشغال شده در حافظه
              



        
