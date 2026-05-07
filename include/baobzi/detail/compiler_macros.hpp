#ifndef BAOBZI_DETAIL_COMPILER_MACROS_HPP
#define BAOBZI_DETAIL_COMPILER_MACROS_HPP

/// Detail-only compiler-attribute helpers. Modeled on polyfit's
/// internal/compiler_macros.h. C++20 is the floor here, so the bare
/// attribute forms are always available; the macros exist to keep call
/// sites readable and to centralise GCC/Clang-only attribute fallbacks.

#if defined(__GNUC__) || defined(__clang__)
#  define BAOBZI_ALWAYS_INLINE [[gnu::always_inline]] inline
#  define BAOBZI_FLATTEN       [[gnu::flatten]]
#else
#  define BAOBZI_ALWAYS_INLINE inline
#  define BAOBZI_FLATTEN
#endif

#endif // BAOBZI_DETAIL_COMPILER_MACROS_HPP
