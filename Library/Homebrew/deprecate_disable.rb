# typed: true
# frozen_string_literal: true

# Helper module for handling `disable!` and `deprecate!`.
#
# @api private
module DeprecateDisable
  module_function

  DEPRECATE_DISABLE_REASONS = {
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
  }.freeze

  def deprecate_disable_info(formula)
    if formula.deprecated?
      type = :deprecated
      reason = formula.deprecation_reason
    elsif formula.disabled?
      type = :disabled
      reason = formula.disable_reason
    else
      return
    end

    reason = DEPRECATE_DISABLE_REASONS[reason] if DEPRECATE_DISABLE_REASONS.key? reason

    [type, reason]
  end
end
