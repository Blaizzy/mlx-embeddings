# frozen_string_literal: true

class MacOSRequirement < Requirement
  module Compat
    def initialize(tags = [], comparator: ">=")
      if tags.first.respond_to?(:map)
        versions, *rest = tags

        versions = versions.map do |v|
          next v if v.is_a?(Symbol)

          sym = MacOS::Version.new(v).to_sym

          odisabled "depends_on macos: #{v.inspect}",
                    "depends_on macos: #{sym.inspect}"
          sym
        end

        tags = [versions, *rest]
      elsif !tags.empty? && !tags.first.is_a?(Symbol)
        v, *rest = tags
        sym = MacOS::Version.new(v).to_sym

        odisabled "depends_on macos: #{v.inspect}",
                  "depends_on macos: #{sym.inspect}"

        tags = [sym, *rest]
      end

      super(tags, comparator: comparator)
    end
  end

  prepend Compat
end
