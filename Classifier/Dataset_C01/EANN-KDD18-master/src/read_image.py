from torchvision import datasets, models, transforms
def read_image():
    image_list = {} # 将处理后的图像对象im存储在image_list字典中，以文件名（不含扩展名）作为键值。
    file_list = ['../data/weibo1/nonrumor_images/', '../data/weibo1/rumor_images/']
    for path in file_list:
        # data_transforms对象，用来处理图像
        data_transforms = transforms.Compose([
            transforms.Resize(324),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif 返回指定路径中的文件和文件夹的名称列表。

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im) # 用来处理图像
                #im = 1
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print(filename)
                # filename.split('/')[-1]：通过分割文件路径，取得文件名。
                # split(".")：通过分割文件名，取得文件名和扩展名。
                # [0]：取得文件名部分。
                # lower()：将文件名转换为小写。
                # 最后将处理后的图像对象im存储在image_list字典中，以文件名（不含扩展名）作为键值。
    print("image length " + str(len(image_list)))
    #print("image names are " + str(image_list.keys()))
    return image_list # 将处理后的图像对象im存储在image_list字典中，以文件名（不含扩展名）作为键值。