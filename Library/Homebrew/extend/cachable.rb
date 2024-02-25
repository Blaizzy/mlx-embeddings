# typed: true
# frozen_string_literal: true

module Cachable
  sig { returns(T::Hash[T.untyped, T.untyped]) }
  def cache
    @cache ||= T.let({}, T.nilable(T::Hash[T.untyped, T.untyped]))
  end

  sig { void }
  def clear_cache
    cache.clear
  end

  # Collect all classes that mix in Cachable so that those caches can be cleared in-between tests.
  if ENV["HOMEBREW_TESTS"]
    def self.included(klass)
      raise ArgumentError, "Don't use Cachable with singleton classes" if klass.singleton_class?

      super if defined?(super)
    end

    # Ignore classes that get inherited from a lot and that have
    # caches that we don't need to clear on the class level.
    IGNORE_INHERITED_CLASSES = %w[Formula Cask].freeze
    private_constant :IGNORE_INHERITED_CLASSES

    def self.extended(klass)
      Registry.list << klass
      klass.extend(Inherited) unless IGNORE_INHERITED_CLASSES.include?(klass.name)
      super if defined?(super)
    end

    module Inherited
      def inherited(klass)
        # A class might inherit Cachable at the instance level
        # and in that case we just want to skip registering it.
        Registry.list << klass if klass.respond_to?(:clear_cache)
        super if defined?(super)
      end
    end

    module Registry
      def self.list
        @list ||= []
      end

      def self.clear_all_caches
        list.each(&:clear_cache)
      end
    end
  end
end
