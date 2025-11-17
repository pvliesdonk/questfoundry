import { type InputHTMLAttributes, forwardRef, useId } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
  fullWidth?: boolean;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, helperText, fullWidth = false, className = '', id, ...props }, ref) => {
    const generatedId = useId();
    const inputId = id ?? generatedId;
    const errorId = error ? `${inputId}-error` : undefined;
    const helperId = !error && helperText ? `${inputId}-helper` : undefined;
    const inputClasses = [
      'px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-primary',
      error ? 'border-red-500 focus:ring-red-500' : 'border-gray-300',
      fullWidth ? 'w-full' : '',
      className,
    ].filter(Boolean).join(' ');

    return (
      <div className={fullWidth ? 'w-full' : ''}>
        {label && (
          <label className="block text-sm font-medium mb-1 text-gray-700" htmlFor={inputId}>
            {label}
            {props.required && <span className="text-red-500 ml-1">*</span>}
          </label>
        )}
        <input
          ref={ref}
          id={inputId}
          className={inputClasses}
          aria-invalid={!!error}
          aria-describedby={errorId ?? helperId}
          {...props}
        />
        {error && (
          <p id={errorId} className="mt-1 text-sm text-red-600">
            {error}
          </p>
        )}
        {!error && helperText && (
          <p id={helperId} className="mt-1 text-sm text-gray-500">
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;
