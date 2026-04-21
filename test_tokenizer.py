from datamodule.transforms import TextTransform
tt = TextTransform()
ids = tt.tokenize("xin chào các bạn")
print(ids)