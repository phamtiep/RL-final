# Multi-Agent Reinforcement Learning (MARL) with QNetwork

## Mô tả dự án

Dự án này triển khai một hệ thống học củng cố đa tác tử (Multi-Agent Reinforcement Learning - MARL) sử dụng mạng nơ-ron QNetwork. Mỗi tác tử trong môi trường có thể quan sát và thực hiện hành động trong môi trường tương ứng, với mục tiêu tối ưu hóa chính sách hành động thông qua việc học từ các trạng thái và phần thưởng nhận được.

Dự án này bao gồm các phần chính:
1. **Mô hình QNetwork**: Mạng nơ-ron sâu được sử dụng để tính toán giá trị Q cho các hành động của tác tử.
2. **Replay Buffer**: Một bộ đệm lưu trữ các chuyển trạng thái để huấn luyện mô hình.
3. **Huấn luyện và Tối ưu hóa**: Quá trình huấn luyện mô hình bằng cách tối ưu hóa hàm mất mát và cập nhật trọng số của mạng nơ-ron.

## Cấu trúc dự án

Dự án có cấu trúc thư mục như sau:

