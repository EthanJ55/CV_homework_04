import face_recognition

val_path = './data/val/'
test_path = './data/test/'
gallery_path = './data/gallery/'
predict = []


def get_val_results():  # 获取正确的验证集分类结果
    results = []
    with open('./data/val_label.txt') as file:
        lines = file.readlines()
        for line in lines:
            results.append(int(line.split()[1]))
    return results


def get_gallery_encodings():  # 获取gallery中图片的encodings
    res = []
    for i in range(50):
        gal_pic = face_recognition.load_image_file('{}{}.jpg'.format(gallery_path, i))
        loc = face_recognition.face_locations(gal_pic, model='cnn')
        temp = face_recognition.face_encodings(gal_pic, known_face_locations=loc)[0]
        res.append(temp)
    return res


def val_recognition(encodings, path):
    # 实时计算准确率
    count_all = 0  # 当前已经识别的图片
    count_true = 0  # 识别结果正确的图片
    results = get_val_results()

    for i in range(2475):
        pic = face_recognition.load_image_file('{}{}.jpg'.format(path, str(i).rjust(6, '0')))
        temp = face_recognition.face_encodings(pic)
        count_all += 1
        # 检测不出来的图直接判断为34号那个带墨镜的贵物
        if len(temp) == 0:
            if results[i] == 34:
                count_true += 1
                print('{} true. The accuracy now is {}.'.format(i, count_true / count_all))
            else:
                print('{} false, the right is {} but result is 34. '
                      'The accuracy now is {}.'.format(i, results[i], count_true / count_all))
            continue
        pic_encoding = temp[0]
        check = face_recognition.face_distance(encodings, pic_encoding)
        # 找到距离最小的图
        min_val = check[0]
        ind = 0
        for j in range(len(check)):
            if check[j] < min_val:
                min_val = check[j]
                ind = j
        if results[i] == ind:
            count_true += 1
            print('{} true. The accuracy now is {}.'.format(i, count_true / count_all))
        else:
            print('{} false, the right is {} but result is {}. '
                  'The accuracy now is {}.'.format(i, results[i], ind, count_true / count_all))


def test_recognition(encodings, path):
    res = []
    for i in range(2475, 4950):
        pic = face_recognition.load_image_file('{}{}.jpg'.format(path, str(i).rjust(6, '0')))
        temp = face_recognition.face_encodings(pic)
        if len(temp) == 0:
            res.append('{}.jpg 34'.format(i))
            print('{} done.'.format(i))
            continue

        # 如果图片中不止一个人，找到gallery中出现的那个人
        person = 0
        distance = 10
        for pic_encoding in temp:
            check = face_recognition.face_distance(encodings, pic_encoding)
            min_val = check[0]
            ind = 0
            for j in range(len(check)):
                if check[j] < min_val:
                    min_val = check[j]
                    ind = j
            if min_val < distance:
                person = ind
                distance = min_val
        res.append('{}.jpg {}'.format(i, person))
        print('{} done.'.format(i))

    # 写入结果
    with open('./data/181250059.txt', 'w') as file:
        for line in res:
            file.write(line + '\n')


if __name__ == '__main__':
    gal_encodings = get_gallery_encodings()
    test_recognition(gal_encodings, test_path)
