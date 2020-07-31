
import glob
import os
import sys
import time
import math
import weakref

try:
    sys.path.append(glob.glob('${ROOT_CARLA}/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import logging
import random

import queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 25)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            print(register_event)
            register_event(q.put)
            #print(q.qsize())
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            if sensor.type_id == "sensor.other.obstacle":
                
                make_queue(sensor.listen)
            else:
            #print(sensor.listen)
                make_queue(sensor.listen)
            #print(len(self._queues))
        return self

    

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout, i) for i,q in enumerate(self._queues)]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout, ind):
        while True:
            #print("sensor",sensor_queue.get(timeout=timeout))
            if ind <= 1:
                data = sensor_queue.get(timeout=timeout)
            else:
                data = sensor_queue.get(timeout=timeout)
            print("data obs",data)
            #print("frame",data.frame)
            if data.frame == self.frame:
                return data


def obs_callback(obs):
    print("Obstacle detected:\n"+str(obs)+'\n')
        #return obs
    


def main():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)

    try:

        world = client.get_world() 

        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.04   # 25fps
        settings.synchronous_mode = True # Enables synchronous mode
        world.apply_settings(settings)

        ego_vehicle = None
        ego_cam = None
        depth_cam = None
        depth_cam02 = None
        sem_cam = None
        rad_ego = None
        lidar_sen = None
        ego_ob = None

        # --------------
        # Query the recording
        # --------------
        
        # Show the most important events in the recording.  
        print(client.show_recorder_file_info("/home/shubham/Carla_scripts/log/RouteScenario_0.log",False))
        # Show actors not moving 1 meter in 10 seconds.  
        print(client.show_recorder_actors_blocked("/home/shubham/Carla_scripts/log/RouteScenario_0.log",1,1))
        # Show collisions between any type of actor.  
        print(client.show_recorder_collisions("/home/shubham/Carla_scripts/log/RouteScenario_0.log",'v','a'))
        

        # --------------
        # Reenact a fragment of the recording
        # --------------
       
        client.replay_file("/home/shubham/Carla_scripts/log/RouteScenario_0.log",0,0,569)
        

        # --------------
        # Set playback simulation conditions
        # --------------
        
        ego_vehicle = world.get_actor(569) #Store the ID from the simulation or query the recording to find out

        # --------------
        # Place spectator on ego spawning
        # --------------
        
        spectator = world.get_spectator()
        world_snapshot = world.wait_for_tick() 
        spectator.set_transform(ego_vehicle.get_transform())
        

        # --------------
        # Change weather conditions
        # --------------
        """
        weather = world.get_weather()
        weather.sun_altitude_angle = -30
        weather.fog_density = 65
        weather.fog_distance = 10
        world.set_weather(weather)
        """


        


        # --------------
        # Add a RGB camera to ego vehicle.
        # --------------
        
        cam_bp = None
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        cam_bp.set_attribute("image_size_x",str(640))
        cam_bp.set_attribute("image_size_y",str(480))
        cam_bp.set_attribute("fov",str(39.32))
        ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        
        

        # --------------
        # Add a Obstacle Detector to ego vehicle.
        # --------------
        
        ob_det = None
        ob_det = world.get_blueprint_library().find('sensor.other.obstacle')
       
        ob_location = carla.Location(2,0,1)
        ob_rotation = carla.Rotation(0,0,0)
        ob_transform = carla.Transform(ob_location,ob_rotation)
        ob_det.set_attribute("distance",str(100))
        ob_det.set_attribute("hit_radius",str(100))
        ob_det.set_attribute("sensor_tick",str(1))
        ob_det.set_attribute("only_dynamics",str(False))
        ob_det.set_attribute("debug_linetrace",str(False))
        ego_ob = world.spawn_actor(ob_det,ob_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        """
        def obs_callback(obs):
            print("Obstacle detected:\n"+str(obs.distance)+'\n')
        ego_ob.listen(lambda obs: obs_callback(obs))
        """

        # --------------
        # Add a Logarithmic Depth camera to ego vehicle. 
        # --------------
        """
        depth_cam = None
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute("image_size_x",str(1920))
        depth_bp.set_attribute("image_size_y",str(1080))
        depth_bp.set_attribute("fov",str(105))
        depth_location = carla.Location(2,0,1)
        depth_rotation = carla.Rotation(0,0,0)
        depth_transform = carla.Transform(depth_location,depth_rotation)
        depth_cam = world.spawn_actor(depth_bp,depth_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        # This time, a color converter is applied to the image, to get the semantic segmentation view
        depth_cam.listen(lambda image: image.save_to_disk('/home/shubham/Carla_scripts/data/de_log/%.6d.jpg' % image.frame,carla.ColorConverter.LogarithmicDepth))
        """
        # --------------
        # Add a Depth camera to ego vehicle. 
        # --------------
        """
        depth_cam02 = None
        depth_bp02 = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp02.set_attribute("image_size_x",str(640))
        depth_bp02.set_attribute("image_size_y",str(480))
        depth_bp02.set_attribute("fov",str(39.32))
        depth_location02 = carla.Location(2,0,1)
        depth_rotation02 = carla.Rotation(0,0,0)
        depth_transform02 = carla.Transform(depth_location02,depth_rotation02)
        depth_cam02 = world.spawn_actor(depth_bp02,depth_transform02,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        # This time, a color converter is applied to the image, to get the semantic segmentation view
        """
        

        # --------------
        # Add a new semantic segmentation camera to ego vehicle
        # --------------
        """
        sem_cam = None
        sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute("image_size_x",str(1920))
        sem_bp.set_attribute("image_size_y",str(1080))
        sem_bp.set_attribute("fov",str(105))
        sem_location = carla.Location(2,0,1)
        sem_rotation = carla.Rotation(0,180,0)
        sem_transform = carla.Transform(sem_location,sem_rotation)
        sem_cam = world.spawn_actor(sem_bp,sem_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        # This time, a color converter is applied to the image, to get the semantic segmentation view
        sem_cam.listen(lambda image: image.save_to_disk('~/tutorial/new_sem_output/%.6d.jpg' % image.frame,carla.ColorConverter.CityScapesPalette))
        """

        # --------------
        # Add a new radar sensor to ego vehicle
        # --------------
        """
        rad_cam = None
        rad_bp = world.get_blueprint_library().find('sensor.other.radar')
        rad_bp.set_attribute('horizontal_fov', str(35))
        rad_bp.set_attribute('vertical_fov', str(20))
        rad_bp.set_attribute('range', str(20))
        rad_location = carla.Location(x=2.8, z=1.0)
        rad_rotation = carla.Rotation(pitch=5)
        rad_transform = carla.Transform(rad_location,rad_rotation)
        rad_ego = world.spawn_actor(rad_bp,rad_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        def rad_callback(radar_data):
            velocity_range = 7.5 # m/s
            current_rot = radar_data.transform.rotation
            for detect in radar_data:
                azi = math.degrees(detect.azimuth)
                alt = math.degrees(detect.altitude)
                # The 0.25 adjusts a bit the distance so the dots can
                # be properly seen
                fw_vec = carla.Vector3D(x=detect.depth - 0.25)
                carla.Transform(
                    carla.Location(),
                    carla.Rotation(
                        pitch=current_rot.pitch + alt,
                        yaw=current_rot.yaw + azi,
                        roll=current_rot.roll)).transform(fw_vec)

                def clamp(min_v, max_v, value):
                    return max(min_v, min(value, max_v))

                norm_velocity = detect.velocity / velocity_range # range [-1, 1]
                r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
                g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
                b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
                world.debug.draw_point(
                    radar_data.transform.location + fw_vec,
                    size=0.075,
                    life_time=0.06,
                    persistent_lines=False,
                    color=carla.Color(r, g, b))
        rad_ego.listen(lambda radar_data: rad_callback(radar_data))
        """

        # --------------
        # Add a new LIDAR sensor to ego vehicle
        # --------------
        """
        lidar_cam = None
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(32))
        lidar_bp.set_attribute('points_per_second',str(90000))
        lidar_bp.set_attribute('rotation_frequency',str(40))
        lidar_bp.set_attribute('range',str(20))
        lidar_location = carla.Location(0,0,2)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        lidar_sen = world.spawn_actor(lidar_bp,lidar_transform,attach_to=ego_vehicle,attachment_type=carla.AttachmentType.Rigid)
        """
        
        with CarlaSyncMode(world, ego_cam, ego_ob,fps=25) as sync_mode:
            while True:

                # Advance the simulation and wait for the data.
                #snapshot, image_rgb, depth_image, point_cloud = sync_mode.tick(timeout=2.0)
                snapshot, im, obs_val = sync_mode.tick(timeout=2.0)
                # Choose the next waypoint and update the car location

                
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                #print("fps",fps)
                print(obs_val)
                #image_rgb.save_to_disk('/home/shubham/Carla_scripts/data/RGB/%.6d.png' % image_rgb.frame)
                #depth_image.save_to_disk('/home/shubham/Carla_scripts/data/Depth/%.6d.png' % depth_image.frame,carla.ColorConverter.Raw)
                #point_cloud.save_to_disk('/home/shubham/Carla_scripts/data/lidar/%.6d.ply' % point_cloud.frame)
    
        """
        while True:
            world_snapshot = world.wait_for_tick()
        """

    finally:
        # --------------
        # Destroy actors
        # --------------
        if ego_vehicle is not None:
            if ego_cam is not None:
                ego_cam.stop()
                ego_cam.destroy()
            if depth_cam is not None:
                depth_cam.stop()
                depth_cam.destroy()
            if sem_cam is not None:
                sem_cam.stop()
                sem_cam.destroy()
            if rad_ego is not None:
                rad_ego.stop()
                rad_ego.destroy()
            if lidar_sen is not None:
                lidar_sen.stop()
                lidar_sen.destroy()
            if ego_ob is not None:
                ego_ob.stop()
                ego_ob.destroy()
            ego_vehicle.destroy()
            
        print('\nNothing to be done.')


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_replay.')
