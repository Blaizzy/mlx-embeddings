# typed: true
# frozen_string_literal: true

class DevelopmentTools
  class << self
    extend T::Sig

    def locate(tool)
      (@locate ||= {}).fetch(tool) do |key|
        @locate[key] = if (path = HOMEBREW_PREFIX/"bin/#{tool}").executable?
          path
        elsif File.executable?(path = "/usr/bin/#{tool}")
          Pathname.new path
        end
      end
    end

    sig { returns(Symbol) }
    def default_compiler
      :gcc
    end
  end
end
