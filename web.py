
def get_model(mode=1):
    opt = TestOptions().parse(save=False)
    opt.display_id = 0 # do not launch visdom
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.in_the_wild = True # This triggers preprocessing of in the wild images in the dataloader
    opt.traverse = True # This tells the model to traverse the latent space between anchor classes
    opt.interp_step = 0.05 # this controls the number of images to interpolate between anchor classes
    data_loader = CreateDataLoader(opt)
    opt.name = 'males_model' if mode else 'females_model' # change to 'females_model' if you're trying the code on a female image  males_model
    model = create_model(opt)
    model.eval()
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    return model, dataset, visualizer

model1, dataset1, visualizer1 = get_model(1)
model0, dataset0, visualizer0 = get_model(0)

def f1(image, mode=1):
    mode = (mode == "男性")
    if mode:
        model, dataset, visualizer = model1, dataset1, visualizer1
    else:
        model, dataset, visualizer = model0, dataset0, visualizer0
    img = Image.fromarray(image)
    img.save('tmp.jpg', 'JPEG')
    data = dataset.dataset.get_item_from_path('tmp.jpg')
    visuals = model.inference(data)
    os.makedirs('results', exist_ok=True)
    visualizer.make_video(visuals, 'tmp1.mp4')
    video_trans_size('tmp1.mp4', 'tmp2.mp4')
    return 'tmp2.mp4'

def f2():
    mp4_path = 'tmp2.mp4'
    output_folder = "output_frames"
    frame_rate = 6
    extract_frames(mp4_path, output_folder, frame_rate)
    images = get_image_paths(output_folder)
    return sorted(images)

def f3():
    res = recognize_faces_in_image("人脸素材", "output_frames", "output_recognize")
    ans = []
    for k, v in res[:3]:
        s = k.split("/")[-1]
        s = s.split(".")[0]
        ans.append(s)
        ans.append(k)
        ans.append(v)
    while len(ans) < 9:
        ans.append(None)
        ans.append(None)
        ans.append(None)
    return ans
    

# 示例图片路径
example_images = [
    ["examples/example_male.jpg", "男性"],
    ["examples/example_female.jpg", "女性"]
]

def load_example(example):
    """辅助函数：加载示例图片和性别选择"""
    img_path, gender = example
    return img_path, gender

with gr.Blocks(css="footer {display: none;}") as demo:
    gr.Markdown("# 寻人系统\n")

    # 定义Step 1中的组件
    with gr.Column():
        gr.Markdown("## STEP 1 面容预测")
        with gr.Row():
            img1 = gr.Image(label="上传您的照片")
            rd = gr.Radio(["男性", "女性"], label="模型选择", value="女性")
            video = gr.Video()
        with gr.Row():
            btn1 = gr.Button("开始预测")

    # 示例模块，仅展示图片供用户选择
    with gr.Row():
        gr.Examples(
            examples=example_images,
            inputs=[img1, rd],  # 将输入绑定到img1和rd
            fn=load_example,  # 辅助函数用于加载示例
            cache_examples=False,  # 禁用缓存
            examples_per_page=2  # 每页显示的示例数量
        )

    # 步骤2：面容提取
    gr.Markdown("## STEP 2 面容提取")
    with gr.Row():
        btn2 = gr.Button("抽帧图片")
    with gr.Row():
        gallery = gr.Gallery(elem_id="gallery", columns=10, height="auto")

    # 步骤3：人脸比对
    gr.Markdown("## STEP 3 人脸比对")
    with gr.Row():
        btn3 = gr.Button("匹配人脸")

    with gr.Tabs():
        with gr.TabItem("第一可能"):
            t1 = gr.Textbox(label="名字")
            with gr.Row():
                i1 = gr.Image(scale=1/5)
                g1 = gr.Gallery(elem_id="g1", columns=20)

        with gr.TabItem("第二可能"):
            t2 = gr.Textbox(label="名字")
            with gr.Row():
                i2 = gr.Image(scale=1/5)
                g2 = gr.Gallery(elem_id="g2", columns=20)

        with gr.TabItem("第三可能"):
            t3 = gr.Textbox(label="名字")
            with gr.Row():
                i3 = gr.Image(scale=1/5)
                g3 = gr.Gallery(elem_id="g3", columns=20)

    # 绑定按钮点击事件
    btn1.click(f1, [img1, rd], video)
    btn2.click(f2, None, gallery)
    btn3.click(f3, None, [t1, i1, g1, t2, i2, g2, t3, i3, g3])

demo.launch(server_name="0.0.0.0", server_port=6006, share=True)