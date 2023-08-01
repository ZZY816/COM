from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .curriculum_center_head import CurriculumCenterHead
from .head_zoo import CurriculumCenterHead_x1, CurriculumCenterHead_x2, CurriculumCenterHead_x3, CurriculumCenterHead_x4, CurriculumCenterHead_car, CurriculumCenterHead_x5, CurriculumCenterHead_x6, CurriculumCenterHead_car_x2, CurriculumCenterHead_x7, CurriculumCenterHead_car_x3, CurriculumCenterHead_x8, CurriculumCenterHead_car_x4, CurriculumCenterHead_ped, CurriculumCenterHead_pc
from .anchor_head_curriculum import AnchorHeadCurriculum
from .head_zoo import CurriculumCenterHead_occupancy, CurriculumCenterHead_distance, CurriculumCenterHead_od, CurriculumCenterHead_odl, CurriculumCenterHead_oda, CurriculumCenterHead_odf
from .curri_anchor_head_single import CurriculumAnchorHeadSingle
from .head_zoo import CurriculumCenterHead_ped_merge, CurriculumAnchorHeadSingle_x1, CurriculumAnchorHeadSingle_car, CurriculumCenterHead_car_merge
from .head_zoo import CurriculumAnchorHeadSingle_car_x2, CurriculumCenterHead_angle, CurriculumCenterHead_length, CurriculumCenterHead_x9

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'CurriculumCenterHead': CurriculumCenterHead,
    'CurriculumCenterHead_x1': CurriculumCenterHead_x1,
    'CurriculumCenterHead_x2': CurriculumCenterHead_x2,
    'CurriculumCenterHead_x3': CurriculumCenterHead_x3,
    'CurriculumCenterHead_x4': CurriculumCenterHead_x4,
    'CurriculumCenterHead_x5': CurriculumCenterHead_x5,
    'CurriculumCenterHead_x6': CurriculumCenterHead_x6,
    'CurriculumCenterHead_x7': CurriculumCenterHead_x7,
    'CurriculumCenterHead_x8': CurriculumCenterHead_x8,
    'CurriculumCenterHead_x9': CurriculumCenterHead_x9,
    'CurriculumCenterHead_car_x2': CurriculumCenterHead_car_x2,
    'CurriculumCenterHead_car_x3': CurriculumCenterHead_car_x3,
    'CurriculumCenterHead_car_x4': CurriculumCenterHead_car_x4,
    'CurriculumCenterHead_car': CurriculumCenterHead_car,
    'CurriculumCenterHead_ped': CurriculumCenterHead_ped,
    'CurriculumCenterHead_ped_merge': CurriculumCenterHead_ped_merge,
    'CurriculumCenterHead_pc': CurriculumCenterHead_pc,
    'CurriculumCenterHead_occupancy': CurriculumCenterHead_occupancy,
    'CurriculumCenterHead_distance': CurriculumCenterHead_distance,
    'CurriculumCenterHead_od': CurriculumCenterHead_od,
    'CurriculumCenterHead_odl': CurriculumCenterHead_odl,
    'CurriculumCenterHead_oda': CurriculumCenterHead_oda,
    'CurriculumCenterHead_odf': CurriculumCenterHead_odf,
    'AnchorHeadCurriculum' : AnchorHeadCurriculum,
    'CurriculumAnchorHeadSingle': CurriculumAnchorHeadSingle,
    'CurriculumAnchorHeadSingle_x1': CurriculumAnchorHeadSingle_x1,
    'CurriculumAnchorHeadSingle_car': CurriculumAnchorHeadSingle_car,
    'CurriculumCenterHead_car_merge': CurriculumCenterHead_car_merge,
    'CurriculumAnchorHeadSingle_car_x2': CurriculumAnchorHeadSingle_car_x2,
    'CurriculumCenterHead_angle': CurriculumCenterHead_angle,
    'CurriculumCenterHead_length': CurriculumCenterHead_length,

}
