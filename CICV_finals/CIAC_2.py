from DataInterfacePython import *
import keyboard
import csv
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from tool import *


def Map_Process(map_csv):
    maps_x_ = []
    maps_y_ = []
    maps_yaw_ = []
    with open(map_csv) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        # print(header_row)
        for row in reader:
            maps_x_.append(float(row[1]))
            maps_y_.append(float(row[2]))
            maps_yaw_.append(float(row[3]))
    return maps_x_, maps_y_, maps_yaw_


"""
1. 全局地图信息获取
"""
dir_root = os.path.dirname(__file__)
map_csv = dir_root + '\CIAC_2_map.csv'
maps_x = Map_Process(map_csv)[0]
maps_y = Map_Process(map_csv)[1]
maps_yaw = Map_Process(map_csv)[2]

maps_s, maps_d = mapxy2frenet(maps_x, maps_y)
ref_s = []
ref_d = []
ref_x = []
ref_y = []
ref_yaw = []

"""
2. 自车变量
"""
ego_state = State()
object_vehicle_state = State()
last_idx = len(maps_x) - 1
"""
3.周围车辆信息变量定义
"""
object_vehicles_ID = []
object_vehicles_x = []
object_vehicles_y = []
object_vehicles_vx = []
object_vehicles_vy = []
object_vehicles_dist = []
object_vehicles_heading = []
object_vehicles_s = []
object_vehicles_d = []
# object_lidar检测到的
object_vehicles_boundary_id = []
object_vehicles_left_id = []
object_vehicles_right_id = []
object_vehicles_middle_id = []
sensor_fusion = []

"""
4.------根据周围信息进行确定
"""
lane_choose = 0
"""
5.信息格式定义
"""
# 主车状态
EGO_STATE_FORMAT = "time@i,x@d,y@d,z@d,yaw@d,pitch@d,roll@d,speed@d"
# 主车控制
EGO_CONTROL_FORMAT = "time@i,valid@b,throttled@d,break@d,steer@d,mode@i,gear@i"
# 车道线检测
LIDAR_FORMAT = "time@i,1024@[,OBJ_ID@i,OBJ_Class@b,OBJ_Shape@i,OBJ_S_X@d,OBJ_S_Y@d,OBJ_S_Z@d,OBJ_S_Dist@d,OBJ_S_Azimuth@d,OBJ_S_Elevation@d,OBJ_Ego_Vx@d,OBJ_Ego_Vy@d,OBJ_Ego_Heading@d,OBJ_Length@d,OBJ_Width@d,OBJ_Height@d"
LANE_INFO_FROMAT = "Timestamp@i,4@[,Lane_ID@i,Lane_Distance@d,Lane_Car_Distance_Left@d,Lane_Car_Distance_Right@d," \
                   "Lane_Curvature@d,Lane_Coefficient_C0@d,Lane_Coefficient_C1@d,Lane_Coefficient_C2@d," \
                   "Lane_Coefficient_C3@d,Lane_Class@b"
"""
6. pid参量
"""
error_last = 0
target = []
ego = []


def ModelStart(userData):
    userData["ego"] = BusAccessor(userData["busId"], "ego", EGO_STATE_FORMAT)

    userData["ego_control"] = BusAccessor(userData["busId"], "ego_control", EGO_CONTROL_FORMAT)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    userData["lidar0"] = BusAccessor(userData["busId"], "Lidar_ObjList_G.0", LIDAR_FORMAT)
    userData["lane_info0"] = BusAccessor(userData["busId"], "LaneInfoPerception.0", LANE_INFO_FROMAT)
    userData["lane_current_type"] = [0, 0, 0, 0]
    userData["lane_current_curvature"] = [0, 0, 0, 0]
    userData["lane_current_closed_dis_left"] = 0
    userData["lane_current_closed_dis_right"] = 0
    userData["is_at_lane_line"] = False  # bool
    userData["current_lane_id"] = None  # int
    userData["START"] = 1
    # TODO 为了控制冗余,车道宽度比实际小一些 3.75
    userData["Lane_width"] = 3.750

    userData["Safety_Margin"] = 20
    userData["Target_Speed"] = 98 / 3.6
    userData["Prediction_ts"] = 5
    userData["T"] = 0.01


def ModelOutput(userData):
    """
        0.全局变量定义
    """
    global error_last
    global target_idx
    global error_d
    global error_yaw
    global max_target_speed
    global error_sum
    global is_at_solid_line
    global super_closed
    global lane_current_curvature_sum
    super_closed = False
    is_at_solid_line = False
    print("    ")
    print("    ")
    print("    ")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    """
        1. 信息获取    
    """
    # print("test-------------------------", len(maps_x), len(maps_y), len(maps_yaw), "START", userData["START"])
    # current_yaw-----弧度制
    ego_time, current_x, current_y, current_z, current_yaw, pitch, roll, current_speed = userData["ego"].readHeader()
    current_yaw_angle = current_yaw * 180 / math.pi
    ts, valid, throttle, brake, steer, mode, gear = userData["ego_control"].readHeader()
    lanewidth = userData["Lane_width"]
    safety_margin = 30
    safety_margin_side = 20
    Prediction_ts = userData["Prediction_ts"]
    T = userData["T"]
    # print("ego_time", ego_time)
    valid = 1
    mode = 1
    gear = 0
    throttle = 0.1
    brake = 0
    steer = 0

    """
        1.1 自车动力学参数获取
    """
    lidar_time, lidar_width = userData["lidar0"].readHeader()
    lane_info_time, lane_info_width = userData["lane_info0"].readHeader()
    # print("lidar_width", lidar_width, "lane_info_width",lane_info_width)
    lane_current_type = userData["lane_current_type"]
    lane_current_curvature = userData["lane_current_curvature"]
    lane_current_closed_dis_left = userData["lane_current_closed_dis_left"]
    lane_current_closed_dis_right = userData["lane_current_closed_dis_right"]
    is_at_lane_line = userData["is_at_lane_line"]  # bool
    target_speed = userData["Target_Speed"]
    target_speed_const = userData["Target_Speed"]

    #  current_lane_id   ___用来判断的
    current_lane_id = userData["current_lane_id"]
    current_lane_id_ = current_lane_id  # 用来进行车道线决策选择
    print("current_x", current_x, "current_y", current_y, "current_speed", current_speed * 3.6, "current_yaw_angle",
          current_yaw_angle)

    # ego—state为vehilce的class。这里对它进行赋值
    ego_state.x = current_x
    ego_state.y = current_y
    ego_state.yaw = current_yaw
    ego_state.v = current_speed
    current_speed_deg = current_speed * 180 / math.pi
    # if里面执行一次
    if userData["START"] == 1:
        # 这里可以实现userData["START"]的全局值的更改
        # if里面执行一次
        # 区域从左往右划分为五部分0-1-2-3-4~~~根据车身所占车道比例（压住实线极为0or4）
        target_idx = 0
        userData["START"] = 0

    current_s, current_d = getFrenet(ego_state, maps_x, maps_y, maps_s)
    # print("maps_s[0]", maps_s[0])
    current_s_head = current_s + 5
    print("current_s", current_s, "current_d", current_d)
    # print("START", ego_state.x, ego_state.y, ego_state.v, ego_state.yaw)

    # 这三行确定了getxy误差满足
    # _current_x, _current_y = getXY(current_s, current_d, maps_s,  maps_x, maps_y)
    # print(_current_x, _current_y)
    # print("error_x", current_x - _current_x, "error_y", current_y - _current_y)

    """
    1.2 目标周围车辆的信息获取
    """
    print("lidar_width--------------------------------------------------------------", lidar_width)
    ### 初始化
    object_vehicles_ID.clear()
    object_vehicles_dist.clear()
    object_vehicles_x.clear()
    object_vehicles_y.clear()
    object_vehicles_vx.clear()
    object_vehicles_vy.clear()
    object_vehicles_s.clear()
    object_vehicles_d.clear()
    object_vehicles_heading.clear()
    object_vehicles_left_id.clear()
    object_vehicles_middle_id.clear()
    object_vehicles_right_id.clear()
    sensor_fusion = [object_vehicles_boundary_id, object_vehicles_left_id, object_vehicles_middle_id,
                     object_vehicles_right_id]
    for i in range(int(lidar_width)):
        OBJ_ID, OBJ_Class, OBJ_Shape, \
        OBJ_S_X, OBJ_S_Y, OBJ_S_Z, OBJ_S_Dist, OBJ_S_Azimuth, OBJ_S_Elevation, \
        OBJ_Ego_Vx, OBJ_Ego_Vy, OBJ_Ego_Heading, OBJ_Length, OBJ_Width, OBJ_Height = userData["lidar0"].readBody(i)
        # 打印周围车辆的信息-------
        # print("OBJ_ID", i, "---", OBJ_ID)
        # print("OBJ_S_Dist", i, "---", OBJ_S_Dist)
        object_vehicles_x.append(OBJ_S_X)
        object_vehicles_y.append(OBJ_S_Y)
        object_vehicles_vx.append(OBJ_Ego_Vx)
        object_vehicles_vy.append(OBJ_Ego_Vy)
        object_vehicles_ID.append(OBJ_ID)
        object_vehicles_dist.append(OBJ_S_Dist)
        object_vehicles_heading.append(OBJ_Ego_Heading)

    ### 在周围存在车的时候执行,不存在的话,这里不进去
    if len(object_vehicles_dist) != 0:
        min_dist_surrounding_car_id = np.argmin(object_vehicles_dist)
        # print("min_dist_surrounding_car_id", min_dist_surrounding_car_id)
        min_dist_surrounding_car_ID = object_vehicles_ID[min_dist_surrounding_car_id]
        # print("min_dist_surrounding_car_ID", min_dist_surrounding_car_ID)
        min_dist_surrounding_car_dist = object_vehicles_dist[min_dist_surrounding_car_id]
        # print("min_dist_surrounding_car_dist", min_dist_surrounding_car_dist)
        """ 周围车辆在雷达(自车)坐标系的坐标转换到全局 """
        object_vehicles_x_world, object_vehicles_y_world = vehilce_to_world(ego_state, object_vehicles_x,
                                                                            object_vehicles_y)
        # 测试了转换到全局坐标系x-y,误差满足
        # print(object_vehicles_x_world[min_dist_surrounding_car_id], object_vehicles_y_world[min_dist_surrounding_car_id])
        # print(current_x,current_y)
        # print("object_vehicles_x_world", object_vehicles_x_world)
        # print("object_vehicles_y_world", object_vehicles_y_world)
        """全局坐标转换到s-d """

        for i in range(int(lidar_width)):
            object_vehicle_x_world = object_vehicles_x_world[i]
            object_vehicle_y_world = object_vehicles_y_world[i]
            object_vehicle_state.x = object_vehicle_x_world
            object_vehicle_state.y = object_vehicle_y_world
            object_vehicle_s, object_vehicle_d = getFrenet(object_vehicle_state, maps_x, maps_y, maps_s)
            # print("object_vehicle_s", object_vehicle_s, "object_vehicle_d",object_vehicle_d)
            # print("error_s", object_vehicle_s - current_s)
            object_vehicles_s.append(object_vehicle_s)
            object_vehicles_d.append(object_vehicle_d)

        # print("object_vehicles_s", object_vehicles_s)
        # print("object_vehicles_d", object_vehicles_d)
        # 已经解决
        # 对检测到的车辆所在的车道进行分类处理 1-2-3
        for i in range(len(object_vehicles_s)):
            if object_vehicles_d[i] > 1.8:
                object_vehicles_left_id.append(i)
            elif object_vehicles_d[i] < -1.8:
                object_vehicles_right_id.append(i)
            else:
                object_vehicles_middle_id.append(i)
        # print("object_vehicles_left_id", object_vehicles_left_id, "object_vehicles_middle_id", object_vehicles_middle_id,
        #       "object_vehicles_right_id", object_vehicles_right_id)
    """
        2. 确定当前车所处的车道位置
        信息:2.1. 所处车道位置 2.2. 是否压线
    """
    for i in range(int(lane_info_width)):
        Lane_ID, Lane_Distance, Lane_Car_Distance_Left, Lane_Car_Distance_Right, Lane_Curvature, \
        _, _, _, _, Lane_Class = userData["lane_info0"].readBody(i)

        # print("index", i)
        # print(userData["lane_info0"].readBody(index=i))
        # print(userData["lane_info0"].readBody(index=i)[-1])
        lane_current_type[i] = Lane_Class
        lane_current_curvature[i] = Lane_Curvature
        # print("Lane_Class", Lane_Class)
        # print("index=",i, "----","Lane_Car_Distance_Left ",Lane_Car_Distance_Left,"Lane_Car_Distance_Right", Lane_Car_Distance_Right)
        if i == 1:
            lane_current_closed_dis_left = Lane_Car_Distance_Left
        if i == 2:
            lane_current_closed_dis_right = Lane_Car_Distance_Right

        lane_current_curvature_average = math.fsum(lane_current_curvature) / 4
        print("平均lane_current_curvature_average", lane_current_curvature_average)
        if current_s < 200:
            target_speed = 87 / 3.6
        if lane_current_curvature_average > 0.001:
            target_speed = 87 / 3.6
        if lane_current_curvature_average > 0.004:
            target_speed = 86 / 3.6

        ## situation1 --压左侧实线
    if lane_current_type[0] == 0:
        if lane_current_closed_dis_left < 0 and lane_current_closed_dis_right > 0:
            is_at_lane_line = True
            current_lane_id = 0
        elif lane_current_closed_dis_left > 0 and lane_current_closed_dis_right > 0:
            is_at_lane_line = False
            current_lane_id = 1
            # todo 考虑没有压线但偏离过大的情况
            if lane_current_closed_dis_right < 0.2 or lane_current_closed_dis_right < 0.2:
                is_at_lane_line = True
        elif lane_current_closed_dis_left > 0 and lane_current_closed_dis_right < 0:
            is_at_lane_line = True
            current_lane_id = 1
    elif lane_current_type[0] == 1:
        if lane_current_closed_dis_left < 0 and lane_current_closed_dis_right > 0:
            is_at_lane_line = True
            current_lane_id = 2
        elif lane_current_closed_dis_left > 0 and lane_current_closed_dis_right > 0:
            is_at_lane_line = False
            current_lane_id = 2
            if lane_current_closed_dis_right < 0.2 or lane_current_closed_dis_right < 0.2:
                is_at_lane_line = True
        elif lane_current_closed_dis_left > 0 and lane_current_closed_dis_right < 0:
            is_at_lane_line = True
            current_lane_id = 2
    elif lane_current_type[0] == 3:
        if lane_current_closed_dis_left < 0 and lane_current_closed_dis_right > 0:
            is_at_lane_line = True
            current_lane_id = 3
        elif lane_current_closed_dis_left > 0 and lane_current_closed_dis_right > 0:
            is_at_lane_line = False
            current_lane_id = 3
            if lane_current_closed_dis_right < 0.2 or lane_current_closed_dis_right < 0.2:
                is_at_lane_line = True
        elif lane_current_closed_dis_left > 0 and lane_current_closed_dis_right < 0:
            is_at_lane_line = True
            current_lane_id = 0

    """ 2.3   确定current_lane_id_来决定变道问题"""
    # todo
    if current_lane_id == None:
        current_lane_id_ = 3  # 合并车道靠右行驶
    elif current_lane_id == 0:
        if current_d > 0:  # 左侧压实线
            current_lane_id_ = 1
        else:
            current_lane_id_ = 3
    else:
        current_lane_id_ = current_lane_id

    if is_at_lane_line:
        if current_lane_id == 0:
            is_at_solid_line = True
        else:
            is_at_solid_line = False

    """
    3. 确定与周围车辆的位置关系
    """
    too_close = False
    prepare_for_lane_change = False
    is_left_lane_free = True
    is_right_lane_free = True
    is_closer_than_safety_margin = False
    is_lane_left_changing = False
    is_lane_right_changing = False
    # 当前车道车辆相对信息
    closed_delta_s_forward = 100000
    closed_delta_s_back = -100000
    closed_delta_s_forward_head = 100000
    closed_delta_s_back_head = -100000
    closed_delta_s_forward_vehicle_id = None
    closed_delta_s_back_vehicle_id = None
    closed_delta_s_forward_vehicle_speed = current_speed
    closed_delta_s_back_vehicle_speed = current_speed
    closed_forward_vehicle_d = 100000
    ## 左车道车辆相对信息
    closed_left_delta_s_forward_vehicle_speed = current_speed
    closed_left_delta_s_back_vehicle_speed = current_speed
    closed_left_delta_s_forward_vehicle_id = None
    closed_left_delta_s_back_vehicle_id = None
    closed_left_delta_s_forward = 1000000
    closed_left_delta_s_back = -1000000

    closed_forward_left_vehicle_d = None
    ## 右车道车辆相对信息
    closed_right_delta_s_forward_vehicle_speed = current_speed
    closed_right_delta_s_back_vehicle_speed = current_speed
    closed_right_delta_s_forward_vehicle_id = None
    closed_right_delta_s_back_vehicle_id = None
    closed_right_delta_s_forward = 1000000
    closed_right_delta_s_back = -1000000
    closed_forward_right_vehicle_d = None

    min_object_vehicles_dist = 10000
    min_object_vehicles_dist_id = None
    min_merge_dist_delta_s = 10000
    min_merge_dist_delta_s_head = 10000
    min_merge_dist_delta_d = 10000
    min_merge_dist_delta_speed = current_speed

    """
        3.1 自车道信息processing
        处理和前车的信息需要用————自车head相对于前车
    """

    if current_lane_id != None:
        for i in range(len(sensor_fusion[current_lane_id_])):
            vehicle_in_ego_road_id = sensor_fusion[current_lane_id_][i]
            # print("vehicle_in_ego_road_id", vehicle_in_ego_road_id)
            delta_s_head = object_vehicles_s[vehicle_in_ego_road_id] - current_s_head
            delta_s = object_vehicles_s[vehicle_in_ego_road_id] - current_s
            delta_s = delta_s + (object_vehicles_vx[vehicle_in_ego_road_id] - current_speed) * Prediction_ts * T

            print("当前车道")
            print("delta_s", delta_s)
            # 前方车辆
            if delta_s > 0:
                if delta_s < closed_delta_s_forward:
                    closed_delta_s_forward = delta_s
                    closed_delta_s_forward_vehicle_id = vehicle_in_ego_road_id
                    closed_delta_s_forward_vehicle_speed = object_vehicles_vx[closed_delta_s_forward_vehicle_id]
            # 后方车辆
            elif delta_s < 0:
                if delta_s > closed_delta_s_back:
                    closed_delta_s_back = delta_s
                    closed_delta_s_back_vehicle_id = vehicle_in_ego_road_id
                    closed_delta_s_back_vehicle_speed = object_vehicles_vx[closed_delta_s_back_vehicle_id]
        closed_delta_s_forward_head = closed_delta_s_forward - 5
        closed_delta_s_back_head = closed_delta_s_back - 5
        if closed_delta_s_forward_vehicle_id != None:
            closed_forward_vehicle_d = object_vehicles_d[closed_delta_s_forward_vehicle_id]
    else:  ## 合并的工况

        min_object_vehicles_dist = np.min(object_vehicles_dist)
        min_object_vehicles_dist_id = np.argmin(object_vehicles_dist)
        min_merge_dist_delta_s_head = object_vehicles_s[min_object_vehicles_dist_id] - current_s_head
        min_merge_dist_delta_s = object_vehicles_s[min_object_vehicles_dist_id] - current_s
        min_merge_dist_delta_d = object_vehicles_d[min_object_vehicles_dist_id] - current_d
        min_merge_dist_delta_speed = object_vehicles_vx[min_object_vehicles_dist_id]

    if current_s > 4600:
        closed_delta_s_forward = 100000

    is_closer_than_safety_margin = (closed_delta_s_forward_head < safety_margin)

    """
        3.2 相邻车道信息处理(仅考虑变道安全性)
    """
    if current_lane_id_ == 1:  # 最左侧--期望右变道
        print("current_lane_id_ == 1 ")
        is_left_lane_free = False
        for i in range(len(sensor_fusion[current_lane_id_ + 1])):
            vehicle_in_ego_road_id = sensor_fusion[current_lane_id_ + 1][i]
            object_vehicle_vx = object_vehicles_vx[vehicle_in_ego_road_id]

            #### delta_s ——————目标车道的每个车加上TTC的delta_s
            delta_s = object_vehicles_s[vehicle_in_ego_road_id] - current_s + (
                    object_vehicles_vx[vehicle_in_ego_road_id] - current_speed) * Prediction_ts * T

            """确定右侧车道最近的车辆信息"""
            if delta_s > 0 and delta_s < closed_right_delta_s_forward:
                closed_right_delta_s_forward = delta_s
                closed_right_delta_s_forward_vehicle_id = vehicle_in_ego_road_id
            elif delta_s < 0 and delta_s > closed_right_delta_s_back:
                closed_right_delta_s_back = delta_s
                closed_right_delta_s_back_vehicle_id = vehicle_in_ego_road_id
            """"""

        if closed_right_delta_s_forward < safety_margin_side / 2 or closed_right_delta_s_back > -3:
            is_right_lane_free = False
        ##### 额外的变道条件
        if closed_right_delta_s_back_vehicle_id != None:
            if closed_right_delta_s_back > -3 and object_vehicles_vx[
                closed_right_delta_s_back_vehicle_id] < current_speed - 0.5 and closed_right_delta_s_back < -2.5:
                is_right_lane_free = True
        print("closed_right_delta_s_forward", closed_right_delta_s_forward, "closed_right_delta_s_back",
              closed_right_delta_s_back)

        if closed_right_delta_s_forward_vehicle_id != None:
            closed_forward_right_vehicle_d = object_vehicles_d[closed_right_delta_s_forward_vehicle_id]
            closed_right_delta_s_forward_vehicle_speed = object_vehicles_vx[closed_right_delta_s_forward_vehicle_id]
            print("closed_right_delta_s_forward_vehicle_speed", closed_right_delta_s_forward_vehicle_speed)
            print("closed_forward_right_vehicle_d", closed_forward_right_vehicle_d)
        if closed_right_delta_s_back_vehicle_id != None:
            closed_right_delta_s_back_vehicle_speed = object_vehicles_vx[closed_right_delta_s_back_vehicle_id]


    elif current_lane_id_ == 3:  # 最右侧--期望左变道
        print("current_lane_id_ == 3")
        is_right_lane_free = False
        for i in range(len(sensor_fusion[current_lane_id_ - 1])):
            vehicle_in_ego_road_id = sensor_fusion[current_lane_id_ - 1][i]

            delta_s = object_vehicles_s[vehicle_in_ego_road_id] - current_s + (
                    object_vehicles_vx[vehicle_in_ego_road_id] - current_speed) * Prediction_ts * T
            """确定左侧车道最近的车辆信息"""
            if delta_s > 0 and delta_s < closed_left_delta_s_forward:
                closed_left_delta_s_forward = delta_s
                closed_left_delta_s_forward_vehicle_id = vehicle_in_ego_road_id
            elif delta_s < 0 and delta_s < closed_left_delta_s_forward:
                closed_left_delta_s_back = delta_s
                closed_left_delta_s_back_vehicle_id = vehicle_in_ego_road_id
            """"""

        if closed_left_delta_s_forward < safety_margin_side / 2 or closed_left_delta_s_back > -3:
            is_left_lane_free = False
        ##### 额外的变道条件
        if closed_left_delta_s_forward_vehicle_id != None:
            if closed_left_delta_s_back > -2.5 and object_vehicles_vx[
                closed_left_delta_s_back_vehicle_id] < current_speed - 0.5 and closed_left_delta_s_back < -2.5:
                is_left_lane_free = True

        if closed_left_delta_s_forward_vehicle_id != None:
            closed_forward_left_vehicle_d = object_vehicles_d[closed_left_delta_s_forward_vehicle_id]
            closed_left_delta_s_forward_vehicle_speed = object_vehicles_vx[closed_left_delta_s_forward_vehicle_id]
        if closed_left_delta_s_back_vehicle_id != None:
            closed_left_delta_s_back_vehicle_speed = object_vehicles_vx[closed_left_delta_s_back_vehicle_id]


    elif current_lane_id_ == 2:
        print("current_lane_id_ == 2")
        # 左侧车道车辆信息提取
        for i in range(len(sensor_fusion[current_lane_id_ - 1])):
            vehicle_in_ego_road_id = sensor_fusion[current_lane_id_ - 1][i]
            delta_s = object_vehicles_s[vehicle_in_ego_road_id] - current_s + (
                    object_vehicles_vx[vehicle_in_ego_road_id] - current_speed) * Prediction_ts * T

            """确定左侧车道最近的车辆信息"""
            if delta_s > 0 and delta_s < closed_left_delta_s_forward:
                closed_left_delta_s_forward = delta_s
                closed_left_delta_s_forward_vehicle_id = vehicle_in_ego_road_id
            elif delta_s < 0 and delta_s < closed_left_delta_s_forward:
                closed_left_delta_s_back = delta_s
                closed_left_delta_s_back_vehicle_id = vehicle_in_ego_road_id
        if closed_left_delta_s_forward < safety_margin_side / 2 or closed_left_delta_s_back > -3:
            is_left_lane_free = False
            ##### 额外的变道条件
        if closed_left_delta_s_forward_vehicle_id != None:
            if closed_left_delta_s_back > -3 and object_vehicles_vx[
                closed_left_delta_s_back_vehicle_id] < current_speed - 0.5 and closed_left_delta_s_back < -2.5:
                is_left_lane_free = True
        if closed_left_delta_s_forward_vehicle_id != None:
            closed_forward_left_vehicle_d = object_vehicles_d[closed_left_delta_s_forward_vehicle_id]
            closed_left_delta_s_forward_vehicle_speed = object_vehicles_vx[closed_left_delta_s_forward_vehicle_id]

        if closed_left_delta_s_back_vehicle_id != None:
            closed_left_delta_s_back_vehicle_speed = object_vehicles_vx[closed_left_delta_s_back_vehicle_id]

        # 右侧车道车辆信息提取--期望右变道
        for i in range(len(sensor_fusion[current_lane_id_ + 1])):
            vehicle_in_ego_road_id = sensor_fusion[current_lane_id_ + 1][i]
            object_vehicle_vx = object_vehicles_vx[vehicle_in_ego_road_id]
            delta_s = object_vehicles_s[vehicle_in_ego_road_id] - current_s + (
                    object_vehicles_vx[vehicle_in_ego_road_id] - current_speed) * Prediction_ts * T

            """确定右侧车道最近的车辆信息"""
            if delta_s > 0 and delta_s < closed_right_delta_s_forward:
                closed_right_delta_s_forward = delta_s
                closed_right_delta_s_forward_vehicle_id = vehicle_in_ego_road_id
            elif delta_s < 0 and delta_s > closed_right_delta_s_back:
                closed_right_delta_s_back = delta_s
                closed_right_delta_s_back_vehicle_id = vehicle_in_ego_road_id
            """"""
        if closed_right_delta_s_forward < safety_margin_side / 2 or closed_right_delta_s_back > -3:
            is_right_lane_free = False
        ##### 额外的变道条件
        if closed_right_delta_s_back_vehicle_id != None:
            if closed_right_delta_s_back > -3 and object_vehicles_vx[
                closed_right_delta_s_back_vehicle_id] < current_speed - 0.5 and closed_right_delta_s_back < -2.5:
                is_right_lane_free = True
        if closed_right_delta_s_forward_vehicle_id != None:
            closed_forward_right_vehicle_d = object_vehicles_d[closed_right_delta_s_forward_vehicle_id]
            closed_right_delta_s_forward_vehicle_speed = object_vehicles_vx[closed_right_delta_s_forward_vehicle_id]

        if closed_right_delta_s_back_vehicle_id != None:
            closed_right_delta_s_back_vehicle_speed = object_vehicles_vx[closed_right_delta_s_back_vehicle_id]
        # 左右变道条件比较
        if is_right_lane_free and is_left_lane_free:
            if closed_left_delta_s_forward > closed_right_delta_s_forward:
                is_right_lane_free = False
            else:
                is_left_lane_free = False

    """
    根据
    """
    if is_closer_than_safety_margin:
        too_close = True
        print("确实too_close")
        # todo  bu ok1

        if closed_delta_s_forward_vehicle_speed + current_speed < 62 / 3.6:
            if current_lane_id_ == 1:
                if closed_right_delta_s_forward - closed_delta_s_forward > safety_margin_side * 1 / 4:
                    print("*********current_lane_id_ == 1", "目标与当前车道前车的车距 ",
                          closed_right_delta_s_forward - closed_delta_s_forward)

                    prepare_for_lane_change = True
            elif current_lane_id_ == 3:
                if closed_left_delta_s_forward - closed_delta_s_forward > safety_margin_side * 1 / 4:
                    print("*******current_lane_id_ == 3", "目标与当前车道前车的车距 ",
                          closed_left_delta_s_forward - closed_delta_s_forward)
                    prepare_for_lane_change = True
            elif current_lane_id_ == 2:

                if closed_right_delta_s_forward - closed_delta_s_forward > safety_margin_side * 1 / 4 and is_right_lane_free:
                    print("*********current_lane_id_ == 2", "目标与当前车道前车的车距 > ",
                          closed_right_delta_s_forward - closed_delta_s_forward)
                    prepare_for_lane_change = True
                elif closed_left_delta_s_forward - closed_delta_s_forward > safety_margin_side * 1 / 4 and is_left_lane_free:
                    print("*******current_lane_id_ == 2", "目标与当前车道前车的车距  > ",
                          closed_left_delta_s_forward - closed_delta_s_forward)
                    prepare_for_lane_change = True
    print("too_close", too_close, "prepare_for_lane_change", prepare_for_lane_change)

    if current_s < 4600:
        if not prepare_for_lane_change:
            is_left_lane_free = False
            is_right_lane_free = False
    print("too_close", too_close, "prepare_for_lane_change", prepare_for_lane_change, "is_left_lane_free",
          is_left_lane_free, "is_right_lane_free", is_right_lane_free)

    """
        4. 横向
    """
    """4.1 横向决策确定车道"""
    # lane_choose 用来确定frenet-d,因此三车道从左到右分别为[1,0,-1]
    # current_lane_id_[1,(1,2,3),3]
    lane_choose = (- current_lane_id_ + 2)
    if current_lane_id_ != 2:
        if prepare_for_lane_change:
            if is_right_lane_free:
                lane_choose = (- current_lane_id_ + 2) - 1
            elif is_left_lane_free:
                lane_choose = (- current_lane_id_ + 2) + 1
            else:
                lane_choose = (- current_lane_id_ + 2)
    if current_lane_id_ == 2:
        if prepare_for_lane_change:
            if is_right_lane_free:
                if not is_lane_left_changing:
                    lane_choose = (- current_lane_id_ + 2) - 1
                    is_lane_right_changing = True
                    print("自车道2正在右变道is_lane_right_changing----", "True")
            elif is_left_lane_free:
                if not is_lane_right_changing:
                    lane_choose = (- current_lane_id_ + 2) + 1
                    is_lane_left_changing = True
                    print("自车道2正在左变道is_lane_left_changing----", "True")
            else:
                lane_choose = (- current_lane_id_ + 2)
                is_lane_right_changing = False
                is_lane_left_changing = False
                print("保持当前车中is_lane_right_changing and is_lane_left_changing-----", "False")

    if current_s > 4590:
        print("%%%%%%%%%%%%current_lane_id_", current_lane_id_)
        if current_lane_id_ == 1:
            if is_right_lane_free == True:
                lane_choose = 0
            else:
                lane_choose = 1
        elif current_lane_id_ == 2:

            lane_choose = 0
        elif current_lane_id_ == 3:
            lane_choose = -1
        elif current_lane_id_ == 0:
            if current_d > 0:
                lane_choose = 1
            elif current_d < 0:
                lane_choose = -1

    print("lane_choose", lane_choose)
    error_d = current_d - (lane_choose * lanewidth)
    error_d = math.fabs(error_d)
    """4.2 stanley_control -----横向控制"""

    ref_x.clear()
    ref_y.clear()
    ref_s = [current_s, current_s + 25, current_s + 30, current_s + 40, current_s + 50]
    ref_d = [current_d, lane_choose * lanewidth, lane_choose * lanewidth, lane_choose * lanewidth,
             lane_choose * lanewidth]

    ### SD 转换为 XY
    for i in range(len(ref_s)):
        s = ref_s[i]

        d = ref_d[i]
        x, y = getXY(s, d, maps_s, maps_x, maps_y)
        ref_x.append(x)
        ref_y.append(y)

    r_x, r_y, r_yaw = [], [], []
    r_x, r_y, r_yaw, r_k, s = calc_spline_course(ref_x, ref_y)

    Steering_control, target_idx, error_yaw = stanley_control(ego_state, r_x, r_y, r_yaw, 0, lane_current_curvature,
                                                              is_at_solid_line, error_d)

    steer = Steering_control / math.pi * 180

    if is_at_lane_line:
        if is_lane_left_changing or is_lane_right_changing:
            steer /= 2
    if closed_delta_s_forward_vehicle_speed + current_speed < 2.5:
        if current_lane_id_ == 1 and lane_choose == 0:
            steer = - 45  # 右转
            print("da右转 ")
        elif current_lane_id_ == 3 and lane_choose == 0:
            steer = 45  # 左转
            print("da左转")

    print("error_yaw, ", error_yaw)
    # if error_yaw
    print("steer", steer)
    """
    5.纵向
    """

    """5.1 纵向决策确定Target_speed"""

    error_yaw_ = math.fabs(error_yaw)
    print("error_d", error_d)
    print("error_yaw_, ", error_yaw_)
    out = 0
    # if current_lane_id != None:
    if too_close:
        # todo 需要处理变道的时候方向盘转角过大的情况！！！
        print("过近了")
        if is_left_lane_free:
            # steer > 0
            if closed_delta_s_forward_vehicle_speed + current_speed > 3.6 and closed_delta_s_forward_head > 2:  # m/s
                if closed_left_delta_s_forward_vehicle_id != None:
                    target_speed = closed_left_delta_s_forward_vehicle_speed + current_speed
                    out = (70 / 3.6 + target_speed) - linear_process(70 / 3.6, target_speed, error_d, lanewidth)
                    target_speed = out
                    print("选用左侧车辆目标速度")
                else:
                    target_speed = target_speed_const / 3.6
                    out = (88 / 3.6 + target_speed) - linear_process(88 / 3.6, target_speed, error_d, lanewidth)
                    target_speed = out
                    print("左侧可变道且无车")
            else:
                print("左侧可变道但前方车速过小")
                target_speed = 50 / 3.6
                out = (65 / 3.6 + target_speed) - linear_process(65 / 3.6, target_speed, error_d, lanewidth)
                target_speed = out

            if closed_delta_s_forward_head < 4:
                print("is_super_closed")
                super_closed = True

        elif is_right_lane_free:
            if closed_delta_s_forward_vehicle_speed + current_speed > 3.6 and closed_delta_s_forward_head > 2:  # m/s
                if closed_right_delta_s_forward_vehicle_id != None:
                    target_speed = closed_right_delta_s_forward_vehicle_speed + current_speed
                    out = (70 / 3.6 + target_speed) - linear_process(70 / 3.6, target_speed, error_d, lanewidth)
                    target_speed = out
                    print("选用右侧车辆目标速度")
                else:
                    target_speed = target_speed_const / 3.6
                    out = (88 / 3.6 + target_speed) - linear_process(88 / 3.6, target_speed, error_d, lanewidth)
                    target_speed = out
                    print("右侧可变道且无车")
            else:
                print("右侧可变道但前方车速过小")
                target_speed = 50 / 3.6
                out = (65 / 3.6 + target_speed) - linear_process(65 / 3.6, target_speed, error_d, lanewidth)
                target_speed = out
            if closed_delta_s_forward_head < 4:
                print("is_super_closed")
                super_closed = True
        else:

            if closed_delta_s_forward_head > safety_margin * 1 / 2:
                out = linear_process(88 / 3.6, closed_delta_s_forward_vehicle_speed + current_speed + 10,
                                     safety_margin - closed_delta_s_forward_head, safety_margin / 2)
                target_speed = out
                print("> /2", out)
            elif closed_delta_s_forward_head > safety_margin / 4:
                out = closed_delta_s_forward_vehicle_speed + current_speed + 4
                target_speed = out
                print("/2 ~ 2/3", out)
            elif closed_delta_s_forward_head > 6:
                out = closed_delta_s_forward_vehicle_speed + current_speed
                target_speed = out
                print(">6", out)
            elif closed_delta_s_forward_head > 3:
                out = closed_delta_s_forward_vehicle_speed + current_speed - 1
                target_speed = out
                print(">3", out)
            else:
                print("超级近！！！")
                super_closed = True
                out = 0
                target_speed = out
                print("< 3", out)

            print("closed_delta_s_forward_vehicle_speed + current_speed",
                  closed_delta_s_forward_vehicle_speed + current_speed)

            print("选用当前车辆目标速度")

    else:

        if error_yaw_ > 0.1 or error_d > 0.9:
            target_speed -= 15 * (error_yaw_)
            steer *= 1.2
            print("减速后1", target_speed)
        elif error_d > 0.8:
            target_speed -= 15 * (error_yaw_)
            steer *= 1.2
            print("减速后2", target_speed)
        elif error_d > 0.6:
            target_speed -= 12 * (error_yaw_)
            print("减速后3", target_speed)
        out = target_speed
    """
    车辆变道加塞的处理
    ----使用他车的d坐标进行处理
    """
    if current_lane_id_ == 1:
        if closed_forward_right_vehicle_d != None:
            if closed_forward_right_vehicle_d > 0.85 and closed_right_delta_s_forward < safety_margin:
                target_speed = closed_right_delta_s_forward_vehicle_speed + current_speed - 0.5
                print("车道1---右侧加塞")
    elif current_lane_id_ == 3:
        if closed_forward_left_vehicle_d != None:
            if closed_forward_left_vehicle_d < -0.85 and closed_forward_left_vehicle_d < safety_margin:
                target_speed = closed_left_delta_s_forward_vehicle_speed + current_speed - 0.5
                print("车道3---左侧加塞")
    elif current_lane_id_ == 2:
        if closed_forward_left_vehicle_d != None:
            if closed_forward_left_vehicle_d < 2.9 and closed_forward_left_vehicle_d < safety_margin:
                target_speed = closed_left_delta_s_forward_vehicle_speed + current_speed - 0.5
                print("车道2---左侧加塞")
        if closed_forward_right_vehicle_d != None:
            if closed_forward_right_vehicle_d > -2.9 and closed_right_delta_s_forward < safety_margin:
                target_speed = closed_right_delta_s_forward_vehicle_speed + current_speed - 0.5
                print("车道2---右加塞")

    if current_s > 60 and current_s < 130:
        print("合并ing！！！")
        print("min_merge_dist_delta_d", min_merge_dist_delta_d)
        if math.fabs(min_merge_dist_delta_d) < 3:
            if min_merge_dist_delta_s_head > 30:
                target_speed = 73 / 3.6
                print("30-", target_speed)
            elif min_merge_dist_delta_s_head > 20:
                out = linear_process(60 / 3.6, min_merge_dist_delta_speed + current_speed,
                                     safety_margin - min_merge_dist_delta_s_head, safety_margin / 2)
                target_speed = out
                print("20-30", out)
            elif min_merge_dist_delta_s_head > 10:
                out = min_merge_dist_delta_speed + current_speed
                target_speed = out
                print("10-20", out)
            else:
                print("0-10 is_super_closed")
                super_closed = True
                out = 0
                target_speed = out
                print("< /4", out)

    print("out", out)
    lane_current_curvature_sum = math.fsum(lane_current_curvature)
    if current_s < 4500 and current_s > 300:
        if lane_current_curvature_sum == 0 and error_yaw_ < 0.1 and error_d < 0.52:
            if current_lane_id_ == 1:
                if closed_delta_s_forward_head > 90:
                    target_speed = 155
                    print("车道1，最舒服的时候")
            if current_lane_id_ == 2:
                if closed_delta_s_forward_head > 90:
                    target_speed = 155
                    print("车道2，最舒服的时候")
                    print("车道2，最舒服的时候")
            if current_lane_id_ == 3:
                if closed_delta_s_forward_head > 90:
                    target_speed = 155
                    print("车道3，最舒服的时候")

    """5.2 pid control -----纵向控制"""
    acceleration_cmd = pid_control(target_speed, ego_state.v, error_last)
    error_last = target_speed - ego_state.v
    print("acceleration_cmd_raw", acceleration_cmd)

    if acceleration_cmd > 0:  # 求节气门开度
        if acceleration_cmd >= f_th(1, current_speed_deg):
            throttle = 1
            brake = 0
        elif acceleration_cmd <= f_th(0, current_speed_deg):
            throttle = 0
            brake = 0
        else:
            for j in np.arange(0, 1, 0.01):
                if abs(acceleration_cmd - f_th(j, current_speed_deg)) <= 0.2:
                    throttle = j
                    brake = 0
    elif acceleration_cmd < 0:  # 求刹车主缸压力
        if acceleration_cmd <= f_br(15, current_speed_deg):
            throttle = 0
            brake = 15
        elif acceleration_cmd >= f_br(0, current_speed_deg):
            throttle = 0
            brake = 0
        else:
            for j in np.arange(0, 15, 0.1):
                if abs(acceleration_cmd - f_br(j, current_speed_deg)) <= 0.2:
                    throttle = 0
                    brake = j
    else:  # 速度保持
        throttle = f_uni(current_speed_deg)
        brake = 0
    print("super_closed", super_closed)
    if super_closed:
        throttle = 0
        brake = 150

    """  使用车之间的相对d进行处理"""
    if current_lane_id_ == 1 and current_s > 200:
        if closed_right_delta_s_forward_vehicle_id != None:
            closed_forward_right_vehicle_d
            print("??????????????????????????????1",
                  math.fabs(closed_forward_right_vehicle_d - current_d))

            if math.fabs(closed_forward_right_vehicle_d - current_d) < 3:
                if closed_right_delta_s_forward - 5 < 5 or (
                        closed_right_delta_s_forward_vehicle_speed < -7 and closed_right_delta_s_forward < safety_margin_side / 2):
                    throttle = 0
                    brake = 2.5
                    print("前车左急变导--纠正")
                    steer *= 3
                elif math.fabs(
                        closed_forward_right_vehicle_d - current_d) < 2.5 and closed_right_delta_s_forward - 5 < 3.6:
                    throttle = 0
                    brake = 150
                    print("前车左急变导--急刹")

    elif current_lane_id_ == 3 and current_s > 200:
        if closed_left_delta_s_forward_vehicle_id != None:

            print("???????????????????????????3",
                  math.fabs(closed_forward_left_vehicle_d - current_d))
            if math.fabs(closed_forward_left_vehicle_d - current_d) < 3:
                if closed_left_delta_s_forward - 5 < 5 or (
                        closed_left_delta_s_forward_vehicle_speed < -7 and closed_left_delta_s_forward < safety_margin_side / 2):
                    throttle = 0
                    brake = 2.5
                    print("前车右急变导--纠正")
                    steer *= 3
                elif math.fabs(
                        closed_forward_left_vehicle_d - current_d) < 2.5 and closed_left_delta_s_forward - 5 < 3.6:
                    throttle = 0
                    brake = 150
                    print("前车右急变导--急刹")
    elif current_lane_id_ == 2 and current_s > 200:
        if closed_right_delta_s_forward_vehicle_id != None:
            print("?????????????????????????2--right",
                  math.fabs(closed_forward_right_vehicle_d - current_d))
            if math.fabs(closed_forward_right_vehicle_d - current_d) < 3:
                if closed_right_delta_s_forward - 5 < 5 or (
                        closed_right_delta_s_forward_vehicle_speed < -7 and closed_right_delta_s_forward < safety_margin_side / 2):
                    throttle = 0
                    brake = 2.5
                    print("前车左急变导--纠正")
                    steer *= 3
                elif math.fabs(
                        closed_forward_right_vehicle_d - current_d) < 2.5 and closed_right_delta_s_forward - 5 < 3.6:
                    throttle = 0
                    brake = 150
                    print("前车左急变导--急刹")
        if closed_left_delta_s_forward_vehicle_id != None:
            print("?????????????????????????2--left",
                  math.fabs(closed_forward_left_vehicle_d - current_d))
            if math.fabs(closed_forward_left_vehicle_d - current_d) < 3:
                if closed_left_delta_s_forward - 5 < 5 or (
                        closed_left_delta_s_forward_vehicle_speed < -7 and closed_left_delta_s_forward < safety_margin_side / 2):
                    throttle = 0
                    brake = 2.5
                    print("前车右急变导--纠正")
                    steer *= 3
                elif math.fabs(
                        closed_forward_left_vehicle_d - current_d) < 2.5 and closed_left_delta_s_forward - 5 < 3.6:
                    throttle = 0
                    brake = 150
                    print("前车右急变导--急刹")

    print("throttle", throttle, "brake", brake)

    print("steer", steer)
    if is_at_solid_line == True:
        if current_d > 0:
            """ 左压实线"""
            if steer > 0:
                steer = -45
        elif current_d < 0:
            """ 右压实线"""
            if steer < 0:
                steer = 45

    userData["ego_control"].writeHeader(
        *(ts, valid, throttle, brake, steer, mode, gear)
    )


def ModelTerminate(userData):
    pass
