# typed: true
# frozen_string_literal: true

# Explicitly prevent `sorbet-runtime` from being loaded.
def gem(name, *)
  raise Gem::LoadError if name == "sorbet-runtime"

  super
end

require "sorbet-runtime-stub"
