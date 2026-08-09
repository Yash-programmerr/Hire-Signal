import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { LoaderCircle } from "lucide-react";
import { cn } from "../../utils/cn";

const variants = cva("focus-ring inline-flex items-center justify-center gap-2 rounded-xl text-sm font-semibold transition-all duration-200 disabled:pointer-events-none disabled:opacity-50", {
  variants: {
    variant: {
      primary: "bg-gradient-to-r from-indigo-500 to-blue-500 text-white shadow-[0_8px_18px_rgba(99,102,241,.24)] hover:-translate-y-0.5 hover:shadow-[0_12px_24px_rgba(99,102,241,.3)]",
      secondary: "bg-indigo-50 text-indigo-700 hover:bg-indigo-100",
      ghost: "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
      outline: "border border-slate-200 bg-white text-slate-700 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700",
    },
    size: { sm: "h-9 px-3.5", md: "h-11 px-5", lg: "h-12 px-6" },
  },
  defaultVariants: { variant: "primary", size: "md" },
});
export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof variants> { loading?: boolean; }
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, loading, children, ...props }, ref) => <button ref={ref} className={cn(variants({ variant, size }), className)} {...props}>{loading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}{children}</button>);
Button.displayName = "Button";
