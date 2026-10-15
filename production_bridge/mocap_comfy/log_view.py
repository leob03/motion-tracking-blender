"""A reusable live-log pane beside the artist's viewport."""
import bpy
from bpy.types import Operator


def is_log_area(area):
    return (area is not None and area.type == 'TEXT_EDITOR'
            and area.spaces.active.text is not None
            and area.spaces.active.text.get('mocap_job_log', False))


def find_log_area(screen):
    return next((area for area in screen.areas if is_log_area(area)), None) if screen else None


def set_log_text(area, text):
    text['mocap_job_log'] = True
    area.type = 'TEXT_EDITOR'
    space = area.spaces.active
    space.text = text
    space.show_word_wrap = True
    space.show_line_numbers = False
    space.show_syntax_highlight = False
    space.show_region_ui = False
    space.show_region_header = True
    area.tag_redraw()


def refresh_log_views(scene, text):
    """An already-open pane follows a newly submitted/resumed job."""
    text['mocap_job_log'] = True
    for window in bpy.context.window_manager.windows:
        if window.scene == scene:
            for area in window.screen.areas:
                if is_log_area(area):
                    set_log_text(area, text)


class MOCAP_OT_live_log(Operator):
    bl_idname = 'mocap_comfy.live_log'
    bl_label = 'Open Live Log'
    bl_description = 'Show the job log beside this viewport; reuse it if already open'

    @classmethod
    def poll(cls, context):
        return (not bpy.app.background and context.window is not None
                and context.area is not None and context.area.type == 'VIEW_3D')

    def execute(self, context):
        text = bpy.data.texts.get(context.scene.mocap_comfy.log_text)
        if text is None:
            self.report({'INFO'}, 'Submit or resume a job to start its live log')
            return {'CANCELLED'}
        text['mocap_job_log'] = True
        area = find_log_area(context.screen)
        if area is None:
            if context.screen.show_fullscreen:
                self.report({'INFO'}, 'Restore the viewport with Ctrl Space before opening a side log')
                return {'CANCELLED'}
            # Blender keeps the original viewport on the left. The new area
            # gets its dimensions at the next redraw, so do not use them here.
            before = {a.as_pointer() for a in context.screen.areas}
            region = next(r for r in context.area.regions if r.type == 'WINDOW')
            with context.temp_override(region=region):
                result = bpy.ops.screen.area_split(direction='VERTICAL', factor=.65)
            if result != {'FINISHED'}:
                return {'CANCELLED'}
            area = next((a for a in context.screen.areas if a.as_pointer() not in before), None)
            if area is None:
                self.report({'WARNING'}, 'There is not enough space to open a side log')
                return {'CANCELLED'}
        set_log_text(area, text)
        return {'FINISHED'}


class MOCAP_OT_close_log(Operator):
    bl_idname = 'mocap_comfy.close_log'
    bl_label = 'Close Live Log'
    bl_description = 'Close this log pane and give its space back to the adjacent editor'

    @classmethod
    def poll(cls, context):
        area = context.area if is_log_area(context.area) else find_log_area(context.screen)
        # Screen operators need a completed redraw after splitting an area.
        return (not bpy.app.background and context.screen is not None
                and not context.screen.show_fullscreen and len(context.screen.areas) > 1
                and area is not None and area.width > 0)

    def execute(self, context):
        area = context.area if is_log_area(context.area) else find_log_area(context.screen)
        region = next(r for r in area.regions if r.type == 'WINDOW')
        with context.temp_override(area=area, region=region):
            return bpy.ops.screen.area_close()


def draw_log_header(self, context):
    if is_log_area(context.area):
        self.layout.operator('mocap_comfy.close_log', text='Close Log', icon='X')
