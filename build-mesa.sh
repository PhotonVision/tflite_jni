git clone https://gitlab.freedesktop.org/mesa/mesa.git mesa_build
cd mesa_build
meson setup build -Dgallium-drivers=rocket -Dvulkan-drivers= -Dteflon=true
meson compile -C build
