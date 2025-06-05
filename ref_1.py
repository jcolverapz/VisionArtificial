import math
class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            center_x = (x + x + w) // 2
            center_y = (y + y + h) // 2
            same_object_detected = False
            for id, pt in self.center_points.items():
  
                if distance < 25:
                    self.center_points[id] = (center_x, center_y)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])     
                    same_object_detected = True
                    break
           if same_object_detected is False:
               self.center_points[self.id_count] = (center_x, center_y)                      
               objects_bbs_ids.append([x, y, w, h, self.id_count])       
               self.id_count += 1
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            var,var,var,var, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
       self.center_points = new_center_points.copy()
       return objects_bbs_ids