from setuptools import find_packages, setup

package_name = 'handeye_calibration'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='li',
    maintainer_email='li@todo.todo',
    description='Camera intrinsic, ArUco pose, and hand-eye calibration tools for SI190C',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_capture = handeye_calibration.image_capture:main',
            'camera_calibration = handeye_calibration.camera_calibration:main',
            'aruco_detector = handeye_calibration.aruco_detector:main',
            'data_collector = handeye_calibration.data_collector:main',
            'handeye_solver = handeye_calibration.handeye_solver:main',
            'verifier = handeye_calibration.verifier:main',
            'calibrate = handeye_calibration.calibrate:main',
        ],
    },
)
