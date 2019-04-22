# frozen_string_literal: true

module Cachable
  def cache
    @cache ||= {}
  end

  def clear_cache
    cache.clear
  end
end
