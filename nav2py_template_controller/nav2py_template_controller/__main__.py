import yaml
import nav2py
import nav2py.interfaces


class nav2py_template_controller(nav2py.interfaces.nav2py_costmap_controller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_callback('path', self._path_callback)

    def _path_callback(
        self,
        path_: bytes,
    ):
        path = yaml.safe_load(path_.decode())
        linear_x = 1
        angular_v = 0
        self._send_cmd_vel(linear_x, angular_v)


if __name__ == "__main__":
    nav2py.main(nav2py_template_controller)
