# typed: true
# frozen_string_literal: true

require "resource"

module Language
  # Helper functions for Go formulae.
  #
  # @api public
  module Go
    # Given a set of resources, stages them to a gopath for
    # building Go software.
    # The resource names should be the import name of the package,
    # e.g. `resource "github.com/foo/bar"`.
    def self.stage_deps(resources, target)
      if resources.empty?
        if Homebrew::EnvConfig.developer?
          odie "Tried to stage empty Language::Go resources array"
        else
          opoo "Tried to stage empty Language::Go resources array"
        end
      end
      resources.grep(Resource::Go) { |resource| resource.stage(target) }
    end
  end
end
