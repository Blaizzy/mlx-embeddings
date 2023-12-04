# typed: true
# frozen_string_literal: true

# Helper module for handling `disable!` and `deprecate!`.
#
# @api private
module DeprecateDisable
  module_function

  SHARED_DEPRECATE_DISABLE_REASONS = {
    repo_archived: "has an archived upstream repository",
    repo_removed:  "has a removed upstream repository",
  }.freeze

  FORMULA_DEPRECATE_DISABLE_REASONS = {
    does_not_build:      "does not build",
    no_license:          "has no license",
    repo_archived:       "has an archived upstream repository",
    repo_removed:        "has a removed upstream repository",
    unmaintained:        "is not maintained upstream",
    unsupported:         "is not supported upstream",
    deprecated_upstream: "is deprecated upstream",
    versioned_formula:   "is a versioned formula",
    checksum_mismatch:   "was built with an initially released source file that had " \
                         "a different checksum than the current one. " \
                         "Upstream's repository might have been compromised. " \
                         "We can re-package this once upstream has confirmed that they retagged their release",
    **SHARED_DEPRECATE_DISABLE_REASONS,
  }.freeze

  CASK_DEPRECATE_DISABLE_REASONS = {
    discontinued:      "is discontinued upstream",
    unsigned_artifact: "has an unsigned binary which prevents it from running on Apple Silicon devices " \
                       "under standard macOS security policy",
    **SHARED_DEPRECATE_DISABLE_REASONS,
  }.freeze

  def deprecate_disable_info(formula_or_cask)
    if formula_or_cask.deprecated?
      type = :deprecated
      reason = formula_or_cask.deprecation_reason
    elsif formula_or_cask.disabled?
      type = :disabled
      reason = formula_or_cask.disable_reason
    else
      return
    end

    reason = if formula_or_cask.is_a?(Formula) && FORMULA_DEPRECATE_DISABLE_REASONS.key?(reason)
      FORMULA_DEPRECATE_DISABLE_REASONS[reason]
    elsif formula_or_cask.is_a?(Cask::Cask) && CASK_DEPRECATE_DISABLE_REASONS.key?(reason)
      CASK_DEPRECATE_DISABLE_REASONS[reason]
    end

    [type, reason]
  end
end
