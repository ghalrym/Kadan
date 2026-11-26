import path from 'node:path'
import type { AtRule, Plugin, Rule } from 'postcss'

/** Prefix every rule in src/<section>/ with html[data-app="<section>"].
 *
 * Each section was a standalone app with its own global stylesheet. Scoping lets
 * all three load together; the shell sets data-app on <html> for the open page.
 */
export function scopeSectionCss(sourceDirectory: string, sections: readonly string[]): Plugin {
  const sectionOf = (file: string | undefined) => {
    if (!file) return null
    const [first, ...rest] = path.relative(sourceDirectory, file).split(path.sep)
    return rest.length && sections.includes(first) ? first : null
  }
  return {
    postcssPlugin: 'scope-section-css',
    Once(root) {
      root.walkRules((rule: Rule) => {
        const parent = rule.parent as AtRule | undefined
        if (parent?.type === 'atrule' && /keyframes$/i.test(parent.name)) return
        const section = sectionOf(rule.source?.input.file ?? root.source?.input.file)
        if (!section) return
        const scope = `html[data-app="${section}"]`
        rule.selectors = rule.selectors.flatMap((selector) => scopeSelector(selector.trim(), scope))
      })
    },
  }
}

/** Selectors that targeted the document root attach to it; the rest become descendants. */
export function scopeSelector(selector: string, scope: string): string[] {
  const root = /^(?:html|:root)(?![\w-])/.exec(selector)
  if (root) return [scope + selector.slice(root[0].length)]
  // Tsuchi's theme attribute sits on <html>.
  if (selector.startsWith('[data-theme')) return [scope + selector]
  // Pseudo-elements such as ::view-transition-* and the page scrollbar belong to <html>.
  if (selector.startsWith('::')) return [scope + selector, `${scope} ${selector}`]
  return [`${scope} ${selector}`]
}
