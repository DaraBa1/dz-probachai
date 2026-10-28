print("=== ЗАДАЧА 1: Один крок для Datsun PL510 ===")
weight_car = 0.97
target_y = 8.7
model_w = 5.0
learning_rate = 0.1

prediction = model_w * weight_car
error = (prediction - target_y) ** 2
print(f"Початковий прогноз: {prediction:.4f} | Помилка: {error:.4f}")

gradient = 2 * (prediction - target_y) * weight_car
print(f"Крутизна (градієнт): {gradient:.4f}")

model_w_updated = model_w - learning_rate * gradient
print(f"Оновлена вага w: {model_w_updated:.4f}")

new_prediction = model_w_updated * weight_car
new_error = (new_prediction - target_y) ** 2
print(f"Новий прогноз: {new_prediction:.4f} | Зменшена помилка: {new_error:.4f}\n")


print("=== ЗАДАЧА 2: Порівняння для різних машин ===")
vehicles = [
    {"name": "Fiat 128", "x": 0.85, "y": 8.1},
    {"name": "Pontiac Catalina", "x": 2.01, "y": 16.8}
]

for w_val in [5, 9]:
    print(f"--- Розрахунок при w = {w_val} ---")
    for car in vehicles:
        pred_val = w_val * car["x"]
        grad_val = 2 * (pred_val - car["y"]) * car["x"]
        print(f"  {car['name']}: прогноз = {pred_val:.2f}, крутизна = {grad_val:.4f}")
print()


print("=== ЗАДАЧА 3: Дослідження розміру кроку (Learning Rate) ===")
initial_w = 12
step_size = 6
current_w = initial_w

for iteration in range(1, 4):
    grad_val = 2 * (current_w - 8)
    current_w = current_w - step_size * grad_val
    current_err = (current_w - 8) ** 2
    print(f"Крок {iteration}: w = {current_w:.2f}, помилка = {current_err:.2f}")
print("Висновок: Занадто великий lr призводить до розходження моделі.\n")


print("=== ЗАДАЧА 4: Зворотне поширення (Backprop) ===")
x_val = 1.0
y_val = 6.0
w1_val = 2.0
w2_val = 2.0
lr_bp = 0.05

hidden = x_val * w1_val
final_pred = hidden * w2_val
base_error = (final_pred - y_val) ** 2
print(f"Прихований шар h = {hidden}, Прогноз = {final_pred}, Помилка = {base_error}")

d_pred = 2 * (final_pred - y_val)
grad_w2 = d_pred * hidden
grad_w1 = d_pred * w2_val * x_val
print(f"Градієнти: по w2 = {grad_w2}, по w1 = {grad_w1}")

w1_val -= lr_bp * grad_w1
w2_val -= lr_bp * grad_w2
new_hidden = x_val * w1_val
new_final_pred = new_hidden * w2_val
new_bp_error = (new_final_pred - y_val) ** 2
print(f"Після кроку навчання: w1 = {w1_val:.2f}, w2 = {w2_val:.2f}, нова помилка = {new_bp_error:.4f}\n")


print("=== ЗАДАЧА 5*: Нульові ваги на старті ===")
w1_zero = 0.0
w2_zero = 0.0
h_zero = x_val * w1_zero
pred_zero = h_zero * w2_zero
err_zero = (pred_zero - y_val) ** 2

d_pred_zero = 2 * (pred_zero - y_val)
g_w2_zero = d_pred_zero * h_zero
g_w1_zero = d_pred_zero * w2_zero * x_val

print(f"Початкова помилка: {err_zero}")
print(f"Градієнти при нулях: grad_w1 = {g_w1_zero}, grad_w2 = {g_w2_zero}")
print("Висновок: Навчання не відбувається, бо градієнти дорівнюють нулю.")