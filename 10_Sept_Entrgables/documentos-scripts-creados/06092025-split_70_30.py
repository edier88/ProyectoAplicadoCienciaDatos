split_70_30.py 06092025

Cambiar carpetas de entrada/salida:
python split_70_30.py --input resultados_agrupados --out-train train-70 --out-test test-30

Si por error algún archivo trae fechas repetidas y quieres consolidarlas automáticamente:
python split_70_30.py --aggregate-on-duplicates


Cambiar el nombre de la columna de fecha (si fuese distinto):
python split_70_30.py --date-col FECHA.CONEXION

