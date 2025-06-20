from zhipuai import ZhipuAI
import base64
import os

# 初始化客户端
client = ZhipuAI(api_key="1aac81e5b6ad4c0ba96435e85aa36f95.7knscmqgqjzUm6Ds")

def image_to_base64(image_path):
    """
    将图片文件转换为 base64 编码字符串，并自动识别格式（PNG/JPG）
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 检查文件扩展名是否合法
    valid_extensions = ['.jpg', '.jpeg', '.png']
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in valid_extensions:
        raise ValueError(f"不支持的图片格式: {file_ext}. 仅支持: {valid_extensions}")

    # 读取图片并编码为 base64
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode('utf-8')
    
    # 根据图片格式设置 MIME 类型
    if file_ext in ['.jpg', '.jpeg']:
        mime_type = "image/jpeg"
    else:  # .png
        mime_type = "image/png"
    
    return base64_str, mime_type

def analyze_single_image(image_path, index):
    """
    分析单张图片并返回格式化结果
    """
    # 1. 将图片转为 base64 和 MIME 类型
    image_base64, mime_type = image_to_base64(image_path)
    
    # 2. 构造多模态请求
    response = client.chat.completions.create(
        model="glm-4v",  # 多模态模型
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""请分析以下图片，并以'图片内容分析'的格式返回结果。
                                如果图片与电影无关则返回'与电影无关联，无有效信息'"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        stream=False  # 非流式响应（直接返回完整结果）
    )
    
    # 3. 返回格式化后的分析结果
    return f"第{index}张照片，图片分析内容: {response.choices[0].message.content}"

def analyze_multiple_images(image_paths):
    """
    分析多张图片（最多10张），并返回每张图片的分析结果
    """
    # 检查图片数量是否超过限制
    if len(image_paths) > 10:
        raise ValueError("图片数量不能超过10张")
    
    # 存储所有图片的分析结果
    all_results = []
    
    # 逐个分析图片
    for index, image_path in enumerate(image_paths, start=1):
        try:
            print(f"正在分析第 {index} 张图片: {image_path}")
            result = analyze_single_image(image_path, index)
            all_results.append(result)
        except Exception as e:
            error_msg = f"第{index}张照片分析失败: {str(e)}"
            all_results.append(error_msg)
            print(error_msg)
    
    return all_results



"""
# --- 使用示例 ---
if __name__ == "__main__":
    # 替换为你的图片路径列表（支持 .jpg 或 .png）
    image_paths = [
        "E:\\LLMmodel\\Movie-Q-A\\images\\chat_preview.png",
         "E:\\LLMmodel\\Movie-Q-A\\images\\chat_preview.png",
         "C:\\Users\\C\\Desktop\\5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG.jpg"
        # 可以添加更多图片路径，最多10张
        # "E:\\LLMmodel\\Movie-Q-A\\images\\another_image.jpg",
    ]
    
    try:
        # 分析多张图片
        results = analyze_multiple_images(image_paths)
        
        # 打印所有结果
        print("\n===== 图片分析结果汇总 =====")
        for result in results:
            print(result)
            print("-" * 40)  # 分隔线
        
    except Exception as e:
        print(f"发生错误: {e}")"""