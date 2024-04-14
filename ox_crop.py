from PIL import Image

# inputimg_dir = "/home/min/pytorch-ox/test_image/20240413_023659.jpg"

def imgcut(input_dir):
# 이미지 불러오기
    img = Image.open(input_dir)
    # 이미지 회전하기
    img = img.rotate(270)

    imgresized = img.resize((400,400))

    # (left, top, right, bottom) 영역을 지정하여 자르기
    left =[111,180,250,110,176,250,102,176,253]
    top = [112,180,250]
    img1 = imgresized.crop((left[0],top[0],left[0]+35,top[0]+35)) #image1
    img2 = imgresized.crop((left[1],top[0],left[1]+35,top[0]+35)) #image2
    img3 = imgresized.crop((left[2],top[0],left[2]+35,top[0]+35)) #image3
    img4 = imgresized.crop((left[3],top[1],left[3]+30,top[1]+30)) #image4
    img5 = imgresized.crop((left[4],top[1]-4,left[4]+40,top[1]+36)) #image5
    img6 = imgresized.crop((left[5],top[1]-4,left[5]+40,top[1]+36)) #image6
    img7 = imgresized.crop((left[6],top[2]-5,left[6]+40,top[2]+35)) #image7
    img8 = imgresized.crop((left[7],top[2]-5,left[7]+42,top[2]+35)) #image8
    img9 = imgresized.crop((left[8],top[2]-5,left[8]+42,top[2]+35)) #image9

    # img9.show()
    # 이미지 저장하기(jpeg)
    img1.save('/home/min/pytorch-ox/infer_dataset/82.jpg')
    img2.save('/home/min/pytorch-ox/infer_dataset/83.jpg')
    img3.save('/home/min/pytorch-ox/infer_dataset/84.jpg')
    img4.save('/home/min/pytorch-ox/infer_dataset/85.jpg')
    img5.save('/home/min/pytorch-ox/infer_dataset/86.jpg')
    img6.save('/home/min/pytorch-ox/infer_dataset/87.jpg')
    img7.save('/home/min/pytorch-ox/infer_dataset/88.jpg')
    img8.save('/home/min/pytorch-ox/infer_dataset/89.jpg')
    img9.save('/home/min/pytorch-ox/infer_dataset/90.jpg')

# if __name__ == '__main__':
#     main(inputimg_dir)