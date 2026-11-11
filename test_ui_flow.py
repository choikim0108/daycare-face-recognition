"""Real Tk widgets and generated color images; face inference is mocked."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import app
from test_review_flow import face


class UIFlowTests(unittest.TestCase):
    def test_classify_correct_and_explicitly_approve(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            refs, inputs, results = [root / name for name in ('refs', 'input', 'results')]
            for path in (refs / 'Alice', refs / 'Bob', inputs, results):
                path.mkdir(parents=True)
            Image.new('RGB', (160, 160), 'blue').save(refs / 'Alice' / 'base.jpg')
            Image.new('RGB', (160, 160), 'green').save(inputs / 'new.jpg')
            ui = app.App()
            try:
                ui.profile_dir, ui.input_dir, ui.output_dir = map(str, (refs, inputs, results))
                with patch.object(app.face_engine, 'represent', return_value=[face([1, 0])]), \
                     patch.object(app.messagebox, 'showinfo'), \
                     patch.object(app.messagebox, 'showerror') as error, \
                     patch.object(app.messagebox, 'askyesno', return_value=True):
                    deadline = time.monotonic() + 15
                    def wait_done():
                        if not ui.busy or time.monotonic() > deadline:
                            ui.quit()
                        else:
                            ui.after(25, wait_done)
                    ui.after(50, ui.btn_start.invoke)
                    ui.after(150, wait_done)
                    ui.mainloop()
                    self.assertFalse(ui.busy, 'classification timed out')
                    ui.update()
                    self.assertTrue((results / 'Alice' / 'new.jpg').exists())
                    self.assertFalse((refs / 'Alice' / 'new.jpg').exists())
                    ui.child_var.set('Alice')
                    ui.load_child_images('Alice')
                    ui.update()
                    self.assertEqual(len(ui.image_widgets), 1)
                    ui.toggle_image_selection(str(results / 'Alice' / 'new.jpg'))
                    ui.target_var.set('Bob')
                    ui.btn_move.invoke()
                    self.assertTrue((results / 'Bob' / 'new.jpg').exists())
                    self.assertFalse((refs / 'Bob' / 'new.jpg').exists())
                    ui.child_var.set('Bob')
                    ui.load_child_images('Bob')
                    ui.toggle_image_selection(str(results / 'Bob' / 'new.jpg'))
                    ui.approve_selected_images()
                    self.assertTrue((refs / 'Bob' / 'new.jpg').exists())
                    self.assertTrue((inputs / 'new.jpg').exists())
                    self.assertTrue((results / 'Bob' / 'new.jpg').exists())
                    error.assert_not_called()
            finally:
                ui.destroy()


if __name__ == '__main__':
    unittest.main(verbosity=2)
