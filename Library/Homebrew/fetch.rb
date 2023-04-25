# typed: strict
# frozen_string_literal: true

module Homebrew
  # @api private
  module Fetch
    sig { params(formula: Formula, args: CLI::Args).returns(T::Boolean) }
    def fetch_bottle?(formula, args:)
      bottle = formula.bottle

      return true if args.force_bottle? && bottle.present?
      return true if args.bottle_tag.present? && formula.bottled?(args.bottle_tag)

      bottle.present? &&
        formula.pour_bottle? &&
        args.build_from_source_formulae.exclude?(formula.full_name) &&
        bottle.compatible_locations?
    end
  end
end
