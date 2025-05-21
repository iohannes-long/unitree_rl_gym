class MujocoData:
    @classmethod
    def get_position(self, d, body_name):
        body_posi = d.body(body_name).xpos
        return body_posi
    
    @classmethod
    def get_quat(self, d, body_name):
        body_quat = d.body(body_name).xquat
        return body_quat