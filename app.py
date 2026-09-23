import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image

import face_engine

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("어린이집 얼굴 인식 자동 분류기")
        self.geometry("950x750")
        
        self.profile_dir = ""
        self.input_dir = ""
        
        # Tabs
        self.tabview = ctk.CTkTabview(self, width=900, height=700)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)
        
        self.tab_auto = self.tabview.add("자동 분류 (Auto-Sort)")
        self.tab_verify = self.tabview.add("검수 (Verify)")
        
        self.setup_auto_tab()
        self.setup_verify_tab()
        
    def setup_auto_tab(self):
        # 1. Profile Directory Selection
        frame_profile = ctk.CTkFrame(self.tab_auto)
        frame_profile.pack(pady=10, padx=20, fill="x")
        
        self.lbl_profile = ctk.CTkLabel(frame_profile, text="아동 폴더 (Profiles): 선택되지 않음", anchor="w")
        self.lbl_profile.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        btn_profile = ctk.CTkButton(frame_profile, text="폴더 선택", command=self.select_profile_dir)
        btn_profile.pack(side="right", padx=10, pady=10)

        # 2. Input Directory Selection
        frame_input = ctk.CTkFrame(self.tab_auto)
        frame_input.pack(pady=10, padx=20, fill="x")
        
        self.lbl_input = ctk.CTkLabel(frame_input, text="원본 사진 (Input): 선택되지 않음", anchor="w")
        self.lbl_input.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        btn_input = ctk.CTkButton(frame_input, text="폴더 선택", command=self.select_input_dir)
        btn_input.pack(side="right", padx=10, pady=10)
        
        # 3. Strictness Slider
        frame_strict = ctk.CTkFrame(self.tab_auto)
        frame_strict.pack(pady=10, padx=20, fill="x")
        
        self.lbl_strict = ctk.CTkLabel(frame_strict, text="인식 엄격도 (유사도 커트라인): 0.45 (높을수록 깐깐함)", anchor="w")
        self.lbl_strict.pack(side="top", padx=10, pady=5)
        
        self.slider_strict = ctk.CTkSlider(frame_strict, from_=0.2, to=0.8, number_of_steps=60, command=self.update_strictness_label)
        self.slider_strict.set(0.45)
        self.slider_strict.pack(side="top", padx=10, pady=5, fill="x")
        
        # 4. Start Button
        self.btn_start = ctk.CTkButton(self.tab_auto, text="사진 분류 시작", command=self.start_processing, height=50, font=("Helvetica", 16, "bold"))
        self.btn_start.pack(pady=20, padx=20, fill="x")
        
        # 5. Progress
        self.lbl_status = ctk.CTkLabel(self.tab_auto, text="대기 중...", font=("Helvetica", 14))
        self.lbl_status.pack(pady=5)
        
        self.progressbar = ctk.CTkProgressBar(self.tab_auto)
        self.progressbar.pack(pady=5, padx=50, fill="x")
        self.progressbar.set(0)

    def update_strictness_label(self, value):
        self.lbl_strict.configure(text=f"인식 엄격도 (유사도 커트라인): {value:.2f} (높을수록 깐깐함)")

    def setup_verify_tab(self):
        # Top panel for refreshing and selecting child
        top_frame = ctk.CTkFrame(self.tab_verify)
        top_frame.pack(pady=10, padx=10, fill="x")
        
        btn_refresh = ctk.CTkButton(top_frame, text="폴더 새로고침", command=self.refresh_children_list, width=100)
        btn_refresh.pack(side="left", padx=5, pady=10)
        
        self.child_var = ctk.StringVar(value="조회할 폴더 선택")
        self.opt_child = ctk.CTkOptionMenu(top_frame, variable=self.child_var, values=["조회할 폴더 선택"], command=self.load_child_images)
        self.opt_child.pack(side="left", padx=5, pady=10)
        
        # Move target selection
        self.btn_move = ctk.CTkButton(top_frame, text="→ 선택 사진 이동", fg_color="#2B7A0B", hover_color="#1E5607", command=self.move_selected_images, width=120)
        self.btn_move.pack(side="right", padx=5, pady=10)
        
        self.target_var = ctk.StringVar(value="이동할 목적지 선택")
        self.opt_target = ctk.CTkOptionMenu(top_frame, variable=self.target_var, values=["이동할 목적지 선택"])
        self.opt_target.pack(side="right", padx=5, pady=10)

        # Scrollable frame for images
        self.gallery_frame = ctk.CTkScrollableFrame(self.tab_verify)
        self.gallery_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.current_child_images = []
        self.selected_images = set()
        self.image_widgets = {} # img_path -> (label, frame)

    def select_profile_dir(self):
        dir_path = filedialog.askdirectory(title="아동 기준 폴더(Profiles) 선택")
        if dir_path:
            self.profile_dir = dir_path
            self.lbl_profile.configure(text=f"아동 폴더 (Profiles): {self.profile_dir}")
            self.refresh_children_list()

    def select_input_dir(self):
        dir_path = filedialog.askdirectory(title="분류할 원본 사진 폴더 선택")
        if dir_path:
            self.input_dir = dir_path
            self.lbl_input.configure(text=f"원본 사진 (Input): {self.input_dir}")

    def update_progress(self, stage, current, total, message):
        self.lbl_status.configure(text=message)
        if total > 0:
            self.progressbar.set(current / total)
        self.update_idletasks()

    def start_processing(self):
        if not self.profile_dir or not self.input_dir:
            messagebox.showwarning("경고", "아동 폴더와 원본 사진 폴더를 모두 선택해주세요.")
            return
            
        self.btn_start.configure(state="disabled")
        self.progressbar.set(0)
        
        thread = threading.Thread(target=self.processing_task)
        thread.daemon = True
        thread.start()

    def processing_task(self):
        try:
            self.update_progress("profiles", 0, 1, "아동 프로필 학습 준비 중...")
            profiles = face_engine.build_profiles(self.profile_dir, self.update_progress)
            
            if not profiles:
                self.update_progress("error", 0, 1, "학습할 수 있는 아동 프로필이 없습니다.")
                messagebox.showerror("오류", "아동 폴더 내에 사진이 없습니다. 폴더 구조를 확인해주세요.")
                self.btn_start.configure(state="normal")
                return
                
            self.update_progress("process", 0, 1, "사진 분류 시작...")
            threshold = float(self.slider_strict.get())
            face_engine.process_photos(self.input_dir, self.profile_dir, profiles, self.update_progress, threshold)
            
            self.update_progress("done", 1, 1, "작업이 완료되었습니다!")
            messagebox.showinfo("완료", "사진 분류가 완료되었습니다.")
        except Exception as e:
            self.update_progress("error", 0, 1, f"오류 발생: {str(e)}")
            messagebox.showerror("오류", f"처리 중 오류가 발생했습니다:\n{str(e)}")
        finally:
            self.btn_start.configure(state="normal")
            self.after(0, self.refresh_children_list)

    def refresh_children_list(self):
        if not self.profile_dir or not os.path.exists(self.profile_dir):
            return
            
        # 모든 폴더 포함 (미분류 포함)
        children = [f for f in os.listdir(self.profile_dir) if os.path.isdir(os.path.join(self.profile_dir, f))]
        
        if children:
            self.opt_child.configure(values=children)
            
            # 이동 타겟 리스트 (휴지통 추가)
            target_list = [c for c in children] + ["휴지통(영구삭제)"]
            self.opt_target.configure(values=target_list)
            
            current_view = self.child_var.get()
            if current_view == "조회할 폴더 선택" or current_view not in children:
                # 기본으로 첫 번째 폴더 표시
                self.child_var.set(children[0])
                self.load_child_images(children[0])
            else:
                self.load_child_images(current_view)
        else:
            self.opt_child.configure(values=["조회할 폴더 선택"])
            self.child_var.set("조회할 폴더 선택")
            self.opt_target.configure(values=["이동할 목적지 선택"])
            self.clear_gallery()

    def clear_gallery(self):
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
        self.image_widgets.clear()
        self.selected_images.clear()

    def load_child_images(self, child_name):
        self.clear_gallery()
        if child_name == "조회할 폴더 선택":
            return
            
        child_dir = os.path.join(self.profile_dir, child_name)
        if not os.path.exists(child_dir):
            return
            
        images = [f for f in os.listdir(child_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        row = 0
        col = 0
        max_cols = 4
        
        for img_name in images:
            img_path = os.path.join(child_dir, img_name)
            
            try:
                pil_img = Image.open(img_path)
                pil_img.thumbnail((150, 150))
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
                
                frame = ctk.CTkFrame(self.gallery_frame, border_width=0, fg_color="transparent")
                frame.grid(row=row, column=col, padx=10, pady=10)
                
                lbl = ctk.CTkLabel(frame, image=ctk_img, text="")
                lbl.pack(padx=3, pady=3)
                
                lbl.bind("<Button-1>", lambda e, p=img_path: self.toggle_image_selection(p))
                
                self.image_widgets[img_path] = frame
                
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
            except Exception as e:
                print(f"Failed to load {img_name}: {e}")

    def toggle_image_selection(self, img_path):
        frame = self.image_widgets.get(img_path)
        if not frame: return
        
        if img_path in self.selected_images:
            self.selected_images.remove(img_path)
            frame.configure(border_width=0, fg_color="transparent")
        else:
            self.selected_images.add(img_path)
            frame.configure(border_width=3, border_color="red", fg_color="red")

    def move_selected_images(self):
        target = self.target_var.get()
        if not self.selected_images:
            messagebox.showinfo("알림", "이동할 사진을 먼저 클릭해서 선택해주세요.")
            return
        if target == "이동할 목적지 선택":
            messagebox.showinfo("알림", "어디로 이동할지(목적지) 우측 상단에서 선택해주세요.")
            return
            
        if not messagebox.askyesno("확인", f"선택한 {len(self.selected_images)}개의 사진을 '{target}'(으)로 이동하시겠습니까?"):
            return
            
        for img_path in self.selected_images:
            try:
                if target == "휴지통(영구삭제)":
                    os.remove(img_path)
                else:
                    target_dir = os.path.join(self.profile_dir, target)
                    os.makedirs(target_dir, exist_ok=True)
                    filename = os.path.basename(img_path)
                    target_path = os.path.join(target_dir, filename)
                    
                    if os.path.exists(target_path):
                        base, ext = os.path.splitext(filename)
                        target_path = os.path.join(target_dir, f"{base}_moved{ext}")
                        
                    shutil.move(img_path, target_path)
            except Exception as e:
                print(f"Failed to move {img_path}: {e}")
                
        messagebox.showinfo("완료", "이동이 완료되었습니다.")
        self.refresh_children_list()

if __name__ == "__main__":
    app = App()
    app.mainloop()
