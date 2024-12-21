# Multi-Agent Reinforcement Learning (MARL) with QNetwork

## Mô tả dự án

Dự án này triển khai một hệ thống học củng cố đa tác tử (Multi-Agent Reinforcement Learning - MARL) sử dụng mạng nơ-ron QNetwork. Mỗi tác tử trong môi trường có thể quan sát và thực hiện hành động trong môi trường tương ứng, với mục tiêu tối ưu hóa chính sách hành động thông qua việc học từ các trạng thái và phần thưởng nhận được.

Dự án này bao gồm các phần chính:
1. **Mô hình QNetwork**: Mạng nơ-ron sâu được sử dụng để tính toán giá trị Q cho các hành động của tác tử.
2. **Replay Buffer**: Một bộ đệm lưu trữ các chuyển trạng thái để huấn luyện mô hình.
3. **Huấn luyện và Tối ưu hóa**: Quá trình huấn luyện mô hình bằng cách tối ưu hóa hàm mất mát và cập nhật trọng số của mạng nơ-ron.

## Cấu trúc dự án

Dự án có cấu trúc thư mục như sau:

### Mô tả chi tiết các tệp
## Folder models
  - Lưu giữ các tham số cho mô hình huấn luyện dqn-blue-final là tham số cuối cùng
#### 1. **q_network.py**

#### 2. **replay_buffer.py**
   - Tệp này định nghĩa lớp `MultiAgentReplayBuffer`, bộ đệm lưu trữ các chuyển trạng thái (transitions) của các tác tử trong môi trường.

#### 3. **optimize.py**
   - Tệp này chứa các hàm để tối ưu hóa mô hình, bao gồm việc tính toán hàm mất mát và cập nhật trọng số của mô hình.
#### 4. **policy.py**
   - Quyết định mức độ khám phá của mô hình
#### 5. **train.py**
   - Huấn luyện mô hình
#### 6. **myNetwork.py**
   - Định nghĩa mạng neural network điều khiển agent trong mô hình
#### 7. **main.py**
   - Tệp chính để khởi tạo mô hình, bộ đệm, và quá trình huấn luyện. Đây là nơi bạn có thể chạy các thử nghiệm hoặc huấn luyện mô hình trên môi trường MARL.

