import threading
import time
import serial

CAUDAL_MIN = 170
CAUDAL_MAX = 220
PUERTO     = '/dev/ttyACM0'
BAUDRATE   = 9600


class ArduinoController:
    def __init__(self, puerto=PUERTO, baudrate=BAUDRATE):
        self._serial   = None
        self._lock     = threading.Lock()
        self.puerto    = puerto
        self.baudrate  = baudrate

        self.caudal_actual  = float(CAUDAL_MIN)
        self.caudal_sensor  = 0.0
        self.volumen_ml     = 0.0
        self.volumen_litros = 0.0
        self.activo         = False

    # ------------------------------------------------------------------
    # Conexión
    # ------------------------------------------------------------------

    def conectar(self):
        try:
            self._serial = serial.Serial(self.puerto, self.baudrate, timeout=2)
            time.sleep(2)
            self._serial.readline()
            self._serial.readline()
            self._serial.write(b"MODO2\n")
            time.sleep(0.3)
            self._serial.readline()
            print("[Arduino] Conectado en MODO2")
        except Exception as e:
            print(f"[Arduino] Error de conexion: {e}")
            self._serial = None

    def iniciar_hilo_lectura(self):
        if self._serial:
            t = threading.Thread(target=self._leer, daemon=True)
            t.start()

    def cerrar(self):
        self.detener()
        if self._serial:
            self._serial.close()

    @property
    def conectado(self):
        return self._serial is not None

    # ------------------------------------------------------------------
    # Lectura continua (hilo daemon)
    # ------------------------------------------------------------------

    def _leer(self):
        while True:
            try:
                if self._serial and self._serial.in_waiting:
                    resp = self._serial.readline().decode('utf-8', errors='ignore').strip()
                    if resp.startswith("CS="):
                        datos = dict(x.split('=') for x in resp.split(','))
                        with self._lock:
                            self.caudal_sensor  = float(datos.get('CS',  0))
                            self.volumen_ml     = float(datos.get('VOL', 0))
                            self.volumen_litros = float(datos.get('LIT', 0))
            except Exception:
                pass
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Control de bomba
    # ------------------------------------------------------------------

    def iniciar(self):
        if self._serial:
            self._serial.write(b"START\n")
            self.activo = True
            time.sleep(0.5)

    def detener(self):
        if self._serial:
            self._serial.write(b"STOP\n")
            self.activo = False

    def set_caudal(self, caudal):
        caudal = max(CAUDAL_MIN, min(CAUDAL_MAX, caudal))
        if self._serial and self.activo:
            self._serial.write(f"c{caudal}\n".encode())
            self.caudal_actual = caudal
        return caudal

    # ------------------------------------------------------------------
    # Estado (snapshot thread-safe para el overlay)
    # ------------------------------------------------------------------

    def estado(self):
        with self._lock:
            return {
                'activo':         self.activo,
                'caudal_actual':  self.caudal_actual,
                'caudal_sensor':  self.caudal_sensor,
                'volumen_ml':     self.volumen_ml,
                'volumen_litros': self.volumen_litros,
            }
