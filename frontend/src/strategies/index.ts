import type { Strategy, SolveContext, HintResult } from './types'

// Import individual strategies (ordered by difficulty)
import { nakedSingle } from './nakedSingle'
import { hiddenSingle } from './hiddenSingle'
import { cageCombinations } from './killerCageCombinations'
import { inniesOuties } from './inniesOuties'
import { pointingPairs } from './pointingPairs'
import { boxLineReduction } from './boxLineReduction'
import { nakedSubsets } from './nakedSubsets'
import { hiddenSubsets } from './hiddenSubsets'
import { xWing } from './xWing'
import { finnedXWing } from './finnedXWing'
import { swordfish } from './swordfish'
import { skyscraper } from './skyscraper'
import { yWing } from './yWing'
import { xyzWing } from './xyzWing'
import { wWing } from './wWing'
import { wxyzWing } from './wxyzWing'
import { uniqueRectangle } from './uniqueRectangle'
import { bug } from './bug'

// Classic strategies ordered by difficulty (SE rating)
const classicStrategies: Strategy[] = [
  nakedSingle,       // SE 1.0-1.2
  hiddenSingle,      // SE 1.0-1.5
  pointingPairs,     // SE 1.7
  boxLineReduction,  // SE 1.7
  nakedSubsets,      // SE 2.0-3.0
  hiddenSubsets,     // SE 2.3-3.0
  xWing,             // SE 3.2
  finnedXWing,       // SE 3.4
  swordfish,         // SE 3.8
  skyscraper,        // SE 4.0
  yWing,             // SE 4.2
  xyzWing,           // SE 4.4
  wWing,             // SE 4.4
  uniqueRectangle,   // SE 4.5
  wxyzWing,          // SE 4.6
  bug,               // SE 5.0
]

// Killer strategies include cage-specific logic first
const killerStrategies: Strategy[] = [
  nakedSingle,       // SE 1.0-1.2
  hiddenSingle,      // SE 1.0-1.5
  cageCombinations,  // Killer-specific: SE ~1.5
  inniesOuties,      // Killer-specific: SE ~2.0
  pointingPairs,     // SE 1.7
  boxLineReduction,  // SE 1.7
  nakedSubsets,      // SE 2.0-3.0
  hiddenSubsets,     // SE 2.3-3.0
  xWing,             // SE 3.2
  finnedXWing,       // SE 3.4
  swordfish,         // SE 3.8
  skyscraper,        // SE 4.0
  yWing,             // SE 4.2
  xyzWing,           // SE 4.4
  wWing,             // SE 4.4
  uniqueRectangle,   // SE 4.5
  wxyzWing,          // SE 4.6
  bug,               // SE 5.0
]

/**
 * Find a hint by trying strategies in order of difficulty.
 * Returns the first hint found, or null if no hint is available.
 */
export function findHint(context: SolveContext): HintResult | null {
  const strategies = context.gameType === 'killer'
    ? killerStrategies
    : classicStrategies

  for (const strategy of strategies) {
    try {
      const result = strategy.find(context)
      if (result) {
        // For strategies that find placements (like naked/hidden singles),
        // we return them even without eliminations to show the user
        if (result.eliminations.length > 0 || result.highlights.length > 0) {
          return result
        }
      }
    } catch (e) {
      console.error(`Error in strategy ${strategy.name}:`, e)
    }
  }

  return null
}

// Re-export types for convenience
export type { HintResult, SolveContext, Strategy, Elimination, CellHighlight, Cage, Cell } from './types'
