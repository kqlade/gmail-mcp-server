import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: ['dist/**', 'node_modules/**', 'coverage/**'],
  },
  ...tseslint.configs.recommended,
  {
    files: ['src/**/*.ts'],
    rules: {
      // Existing API wrappers intentionally preserve provider-shaped values.
      '@typescript-eslint/no-explicit-any': 'off',
    },
  }
);
