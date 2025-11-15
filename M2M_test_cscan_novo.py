import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import pickle
from perlin_numpy import generate_perlin_noise_2d


class M2k_system:
    def __init__(self):
        self.socket = None
        self.NameOfTheConfiguration = "Config de Simulação"
        self.nb_salvo = 0
        print("Classe M2k_system SIMULADA")

    def set_ip(self, ip, port): pass

    def set_ip_data_server(self, ip, port): pass

    def connect(self):
        self.socket = "dummy_socket"
        return True

    def disconnect(self):
        self.socket = None
        return True

    def update_all_parameters(self, with_focal_laws=False, with_dac_curves=False): pass


class RealTimeCScan:
    def __init__(self, ip="127.0.0.1", remote_port=4444, data_port=4445):
        self.m2m_system = M2k_system()
        self.m2m_system.set_ip(ip, remote_port)
        self.m2m_system.set_ip_data_server(ip, data_port)

        self.cscan_data_points = {}
        self.data_bounds = {
            'min_x': float('inf'), 'max_x': float('-inf'),
            'min_y': float('inf'), 'max_y': float('-inf')
        }

        self.resolution = 0.5
        self.element_pitch = 0.6
        self.num_channels_sim = 64

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.im = None
        self.frame_count = 0
        self.start_time = None
        self.fps_text = None

        self.sim_placa_resolucao = 0.1
        self.sim_placa_largura_mm = 300
        self.sim_placa_altura_mm = 200
        self.sim_placa_offset_x_mm = 0.0
        self.sim_placa_offset_y_mm = 0.0

        largura_px = int(self.sim_placa_largura_mm / self.sim_placa_resolucao)  # 3000
        altura_px = int(self.sim_placa_altura_mm / self.sim_placa_resolucao)  # 2000

        ruido = generate_perlin_noise_2d((altura_px, largura_px), (8, 8))

        espessura_min_corroida = 30.0
        espessura_max_base = 50.0

        self.placa_simulada = np.interp(ruido, (ruido.min(), ruido.max()), (espessura_min_corroida, espessura_max_base))

        self.sim_scan_direction = 1

        centro_probe_offset = (self.num_channels_sim - 1) / 2.0
        metade_abertura_mm = centro_probe_offset * self.element_pitch

        self.current_x_mm = self.sim_placa_offset_x_mm
        self.current_y_mm = self.sim_placa_offset_y_mm + metade_abertura_mm

        self.sim_scan_state = "scanning_x"
        self.sim_scan_speed_mm = 1.0 * self.resolution
        self.sim_scan_step_y_mm = (self.num_channels_sim - 1) * self.element_pitch
        self.sim_target_y_mm = self.current_y_mm
        self.sim_scan_finished = False

    def connect(self):
        print("Conectando ao M2M")
        self.m2m_system.connect()
        print(f"Conexão (simulada) estabelecida!")
        print(f"Configuração: {self.m2m_system.NameOfTheConfiguration}")

    def get_ascan_data(self):
        try:
            ascan_raw = self._sim_get_ascan_data()
            return ascan_raw
        except Exception as e:
            print(f"Erro ao simular A-scan: {e}")
            return None

    def _sim_get_ascan_data(self):
        num_channels = self.num_channels_sim
        num_samples = 512
        ascan_data = np.random.rand(num_channels, num_samples) * 10

        centro_probe_offset = (num_channels - 1) / 2

        for channel_index in range(num_channels):
            channel_offset_mm = (channel_index - centro_probe_offset) * self.element_pitch
            element_x_mm = self.current_x_mm
            element_y_mm = self.current_y_mm + channel_offset_mm

            placa_x_idx = int((element_x_mm - self.sim_placa_offset_x_mm) / self.sim_placa_resolucao)
            placa_y_idx = int((element_y_mm - self.sim_placa_offset_y_mm) / self.sim_placa_resolucao)

            altura, largura = self.placa_simulada.shape
            espessura_lida = 0.0
            if (0 <= placa_y_idx < altura) and (0 <= placa_x_idx < largura):
                espessura_lida = self.placa_simulada[placa_y_idx, placa_x_idx]

            if espessura_lida > 0:
                posicao_pico = int(espessura_lida)
                if 0 < posicao_pico < num_samples - 2:
                    ascan_data[channel_index, posicao_pico - 2: posicao_pico + 2] = 255

        return ascan_data

    def process_ascan_to_cscan(self, ascan_data):
        if ascan_data is None or len(ascan_data) == 0:
            return None
        peak_times = np.argmax(ascan_data, axis=1)
        return peak_times

    def get_encoder_deltas(self):
        if self.sim_scan_finished:
            return 0.0, 0.0

        delta_x = 0.0
        delta_y = 0.0

        min_x_mm = self.sim_placa_offset_x_mm
        max_x_mm = self.sim_placa_offset_x_mm + self.sim_placa_largura_mm
        max_y_mm = self.sim_placa_offset_y_mm + self.sim_placa_altura_mm

        if self.sim_scan_state == "scanning_x":
            delta_x = self.sim_scan_speed_mm * self.sim_scan_direction
            next_x_mm = self.current_x_mm + delta_x

            if next_x_mm >= max_x_mm or next_x_mm < min_x_mm:
                self.sim_scan_state = "sliding_y"
                self.sim_scan_direction *= -1
                self.sim_target_y_mm = self.current_y_mm + self.sim_scan_step_y_mm
                delta_x = 0.0

                if self.sim_target_y_mm >= max_y_mm:
                    self.sim_scan_finished = True

        elif self.sim_scan_state == "sliding_y":
            delta_x = 0.0
            delta_y = self.sim_scan_speed_mm

            if (self.current_y_mm + delta_y) >= self.sim_target_y_mm:
                delta_y = self.sim_target_y_mm - self.current_y_mm
                self.sim_scan_state = "scanning_x"

        return delta_x, delta_y

    def setup_plot(self):
        self.ax.set_title('C-Scan', fontsize=14)
        self.ax.set_xlabel('Posição X (pixels)')
        self.ax.set_ylabel('Posição Y (pixels)')

        lim_x_px = self.sim_placa_largura_mm / self.resolution
        lim_y_px = self.sim_placa_altura_mm / self.resolution

        self.ax.set_xlim(0, lim_x_px)
        self.ax.set_ylim(0, lim_y_px)

        self.fps_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                     verticalalignment='top', color='white',
                                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        plt.tight_layout()

    def update_plot(self, frame):
        delta_x, delta_y = self.get_encoder_deltas()

        if delta_x == 0 and delta_y == 0 and not self.cscan_data_points:
            return []

        self.current_x_mm += delta_x
        self.current_y_mm += delta_y

        ascan_data = self.get_ascan_data()
        if ascan_data is None: return []

        cscan_line = self.process_ascan_to_cscan(ascan_data)
        if cscan_line is None: return []

        new_data_added = False
        num_channels = len(cscan_line)
        centro_probe_offset = (num_channels - 1) / 2

        for channel_index, peak_tof in enumerate(cscan_line):
            channel_offset_mm = (channel_index - centro_probe_offset) * self.element_pitch
            element_x_mm = self.current_x_mm
            element_y_mm = self.current_y_mm + channel_offset_mm

            x_pixel = int(round(element_x_mm / self.resolution))
            y_pixel = int(round(element_y_mm / self.resolution))

            if peak_tof > 0:
                self.cscan_data_points[(x_pixel, y_pixel)] = peak_tof

                self.data_bounds['min_x'] = min(self.data_bounds['min_x'], x_pixel)
                self.data_bounds['max_x'] = max(self.data_bounds['max_x'], x_pixel)
                self.data_bounds['min_y'] = min(self.data_bounds['min_y'], y_pixel)
                self.data_bounds['max_y'] = max(self.data_bounds['max_y'], y_pixel)
                new_data_added = True

        if not new_data_added and self.im is not None:
            elapsed = time.time() - self.start_time
            if elapsed > 0 and self.start_time is not None:
                fps = self.frame_count / elapsed
                self.fps_text.set_text(f'FPS: {fps:.1f}\nPontos: {len(self.cscan_data_points)}')
            return [self.im, self.fps_text]

        if not self.cscan_data_points:
            return []

        width = self.data_bounds['max_x'] - self.data_bounds['min_x'] + 1
        height = self.data_bounds['max_y'] - self.data_bounds['min_y'] + 1
        offset_x = self.data_bounds['min_x']
        offset_y = self.data_bounds['min_y']

        plot_grid = np.full((height, width), 0.0, dtype=np.float32)

        data_items = self.cscan_data_points.items()
        for (px, py), value in data_items:
            idx_x = px - offset_x
            idx_y = py - offset_y
            if (0 <= idx_y < height) and (0 <= idx_x < width):
                plot_grid[idx_y, idx_x] = value

        if self.im is None:
            self.im = self.ax.imshow(plot_grid, aspect='auto',
                                     cmap='jet', interpolation='nearest',
                                     origin='lower', vmin=25, vmax=55)
            plt.colorbar(self.im, ax=self.ax, label='Tempo de Voo / Espessura (mm)')
        else:
            self.im.set_data(plot_grid)
            extent = [
                self.data_bounds['min_x'], self.data_bounds['max_x'],
                self.data_bounds['min_y'], self.data_bounds['max_y']
            ]
            self.im.set_extent(extent)
            self.ax.set_xlim(self.data_bounds['min_x'], self.data_bounds['max_x'])
            self.ax.set_ylim(self.data_bounds['min_y'], self.data_bounds['max_y'])

        self.frame_count += 1
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            self.fps_text.set_text(f'FPS: {fps:.1f}\nPontos: {len(self.cscan_data_points)}')

        return [self.im, self.fps_text]

    def run(self, interval=50):
        try:
            self.connect()
            self.setup_plot()
            time.sleep(1)

            print(f"Intervalo: {interval}ms")
            print(">>> SIMULACAO <<<")

            anim = FuncAnimation(self.fig, self.update_plot,
                                 interval=interval, blit=True, cache_frame_data=False)

            plt.ioff()
            plt.show()

        except KeyboardInterrupt:
            print("\nInterrompido")
        except Exception as e:
            print(f"Erro: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.m2m_system.disconnect()
            print("Desconectado")
            self.save_cscan("cscan_final.pkl")

    def save_cscan(self, filename="cscan_data.pkl"):
        if self.cscan_data_points:
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(self.cscan_data_points, f)
                print(f"\nDados brutos salvos em: {filename}")
                self.save_cscan_image(filename.replace('.pkl', '.png'))
            except Exception as e:
                print(f"Erro ao salvar dados: {e}")

    def save_cscan_image(self, filename="cscan_image.png"):
        print(f"Gerando imagem final: {filename}")
        if not self.cscan_data_points:
            print("Nenhum dado para salvar.")
            return

        width = self.data_bounds['max_x'] - self.data_bounds['min_x'] + 1
        height = self.data_bounds['max_y'] - self.data_bounds['min_y'] + 1
        offset_x = self.data_bounds['min_x']
        offset_y = self.data_bounds['min_y']

        plot_grid = np.full((height, width), 0.0, dtype=np.float32)

        for (px, py), value in self.cscan_data_points.items():
            idx_x = px - offset_x
            idx_y = py - offset_y
            if (0 <= idx_y < height) and (0 <= idx_x < width):
                plot_grid[idx_y, idx_x] = value

        fig_save, ax_save = plt.subplots(figsize=(10, 8))

        im = ax_save.imshow(plot_grid, aspect='auto', cmap='jet',
                            interpolation='nearest', origin='lower',
                            vmin=25, vmax=55)

        im.set_extent([
            self.data_bounds['min_x'], self.data_bounds['max_x'],
            self.data_bounds['min_y'], self.data_bounds['max_y']
        ])

        fig_save.colorbar(im, ax=ax_save, label='Tempo de Voo / Espessura (mm)')
        ax_save.set_title('C-Scan Capturado')
        ax_save.set_xlabel('Posição X (pixels)')
        ax_save.set_ylabel('Posição Y (pixels)')

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Imagem salva em: {filename}")


def main():
    viewer = RealTimeCScan(ip="127.0.0.1", remote_port=4444, data_port=4445)
    viewer.run(interval=10)


if __name__ == '__main__':
    main()