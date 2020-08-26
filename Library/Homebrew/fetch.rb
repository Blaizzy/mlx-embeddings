# frozen_string_literal: true

module Homebrew
  # @api private
  module Fetch
    def fetch_bottle?(f, args:)
      return true if args.force_bottle? && f.bottle
      return false unless f.bottle && f.pour_bottle?
      return false if args.build_from_source_formulae.include?(f.full_name)
      return false unless f.bottle.compatible_cellar?

      true
    end
  end
end
