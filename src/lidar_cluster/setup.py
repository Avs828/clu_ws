from setuptools import find_packages, setup

package_name = 'lidar_cluster'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='park',
    maintainer_email='tjdtn5324@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'lidar_clustering_node = lidar_cluster.lidar_cluster:main',
            'yolov5li_node = lidar_cluster.yolov5li:main',
        ],
    },
)

