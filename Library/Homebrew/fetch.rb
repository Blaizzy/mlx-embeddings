# typed: true
# frozen_string_literal: true

module Homebrew
  # @api private
  module Fetch
    extend T::Sig

    sig { params(f: Formula, args: CLI::Args).returns(T::Boolean) }
    def fetch_bottle?(f, args:)
      bottle = f.bottle

      return true if args.force_bottle? && bottle
      return false unless bottle && f.pour_bottle?
      return false if args.build_from_source_formulae.include?(f.full_name)
      return false unless bottle.compatible_locations?

      true
    end
  end
end
