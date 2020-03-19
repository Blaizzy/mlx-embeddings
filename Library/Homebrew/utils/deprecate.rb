# frozen_string_literal: true

module Utils
  module Deprecate
    # This module is common for both formulae and casks.
    # To-do : Add support for casks + add disable method

    def deprecate
      self.is_deprecated = true

      # deprecate all formulae dependent on this deprecated formula
      all_formulae = ObjectSpace.each_object(Class).select { |klass| klass < Formula }
      all_formulae.each do |f|
        dependencies = f.recursive_dependencies.map(&:name)
        f.is_deprecated = true if (dependencies.include? self.name)
      end
    end

    # Deprecation can be revoked if the underlying problem is fixed
    def revoke_deprecation
      self.is_deprecated = false

      # revoke deprecation from dependents as well
      all_formulae = ObjectSpace.each_object(Class).select { |klass| klass < Formula }
      all_formulae.each do |f|
        dependencies = f.recursive_dependencies.map(&:name)
        revoke = true
        dependencies.each do |d|
          revoke = !(d != self && d.is_deprecated?)
        end
        f.is_deprecated = false if revoke && (dependencies.include? self.name)
      end
    end
  end
end
