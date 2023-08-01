from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .curriculum_center_head import CurriculumCenterHead
from .head_zoo import CurriculumCenterHead_x5
from .anchor_head_curriculum import AnchorHeadCurriculum
from .curri_anchor_head_single import CurriculumAnchorHeadSingle
from .head_zoo import CurriculumCenterHead_ped_merge, CurriculumAnchorHeadSingle_x1, CurriculumAnchorHeadSingle_car, CurriculumCenterHead_car_merge
from .head_zoo import CurriculumAnchorHeadSingle_car_x2

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'CurriculumCenterHead': CurriculumCenterHead,
    'CurriculumCenterHead_x5': CurriculumCenterHead_x5,
    'CurriculumCenterHead_ped_merge': CurriculumCenterHead_ped_merge,
    'AnchorHeadCurriculum' : AnchorHeadCurriculum,
    'CurriculumAnchorHeadSingle': CurriculumAnchorHeadSingle,
    'CurriculumAnchorHeadSingle_x1': CurriculumAnchorHeadSingle_x1,
    'CurriculumAnchorHeadSingle_car': CurriculumAnchorHeadSingle_car,
    'CurriculumCenterHead_car_merge': CurriculumCenterHead_car_merge,
    'CurriculumAnchorHeadSingle_car_x2': CurriculumAnchorHeadSingle_car_x2,
}
