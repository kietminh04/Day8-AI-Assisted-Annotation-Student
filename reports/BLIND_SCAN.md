# Quét độc lập trước khi xem pre-label

Frame: frame_0182.jpg

Số xe nhìn thấy bằng mắt: 20

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. Xe ở làn giữa hướng về phía camera (tọa độ khoảng giữa ảnh, phía dưới): đèn pha cực sáng rọi vệt loang dài trên mặt đường bê tông bóng, AI rất dễ vẽ box ôm cả vệt đèn phản chiếu trên mặt đường thay vì ôm sát thân xe theo GUIDELINE_LABEL.md.
2. Nóc/mui xe tối ở mép dưới cùng ảnh và xe tối chạy cùng chiều ở làn phải phía xa bị che khuất một phần: thân xe chìm vào nền đen, không có đèn phản xạ mạnh nên AI dễ bị False Negative (bỏ sót xe).

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
