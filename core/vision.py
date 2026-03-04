import io
import numpy as np
import onnxruntime as ort
from PIL import Image

ocr_session = ort.InferenceSession("lightweight_ocr_model.onnx")

def extract_text_from_buffer(buffer: io.BytesIO) -> str:
  buffer.seek(0)
  original_img = Image.open(buffer).convert("RGB")

  resized_img = original_img.resize((224, 224))
  img_array = np.array(resized_img, dtype=np.float32)

  img_array /= 255.0
  img_array = np.transpose(img_array, (2, 0 ,1))

  input_tensor = np.expand_dims(img_array, axis=0)

  ocr_result = ocr_session.run(None, {"input": input_tensor})
  return str(ocr_result[0])