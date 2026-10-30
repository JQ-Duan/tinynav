本功能包用于接收传感器数据
sensor_receiver.hpp   用于接收zed mini左右图、深度图与imu数据
imu_buffer.hpp        保存IMU历史
frame_assembler.hpp   按时间戳配对图像间的imu数据
data_preparer.hpp     将对齐后的数据整理成统一格式