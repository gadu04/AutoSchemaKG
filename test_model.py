from openai import OpenAI

# Trỏ client đến server cục bộ của LM Studio
# Bắt buộc phải có base_url, api_key có thể là bất cứ thứ gì
client = OpenAI(
    base_url="http://localhost:1234/v1", 
    api_key="lm-studio"
)

print("Đang gửi yêu cầu tới model, vui lòng chờ...")

completion = client.chat.completions.create(
  # Tên model không quan trọng khi dùng LM Studio,
  # vì nó sẽ tự động dùng model bạn đã tải (load)
  model="local-model/gguf-model-name", 
  messages=[
    {"role": "system", "content": "Bạn là một trợ lý AI hữu ích."},
    {"role": "user", "content": "Viết 3 đoạn IELTS Task 1 với trình độ 9.0."}
  ]
)

# In ra phản hồi từ model GGUF
print(completion.choices[0].message.content)

# ------------------------------------------------------------
# unified header — 2025-12-16.
# ------------------------------------------------------------