# typed: strict
# frozen_string_literal: true

module Homebrew
  # @api private
  module Fetch
    extend T::Sig

    sig { params(f: Formula, args: CLI::Args).returns(T::Boolean) }
    def fetch_bottle?(f, args:)
      bottle = f.bottle

      return true if args.force_bottle? && bottle.present?
      return true if args.bottle_tag.present? && f.bottled?(args.bottle_tag)

      bottle.present? &&
        f.pour_bottle? &&
        args.build_from_source_formulae.exclude?(f.full_name) &&
        bottle.compatible_locations?
    end
  end
end
